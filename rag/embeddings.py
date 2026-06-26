import hashlib
import os
import re
import numpy as np

EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "384"))
_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str):
    return _WORD_RE.findall(text.lower())


class HashingEmbedder:

    name = "hashing-fallback"
    dim = EMBEDDING_DIM

    def embed(self, texts):
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = _tokenize(text)
            if not tokens:
                continue
            # unigrams + bigrams, hashed into fixed-width buckets with sign trick
            grams = list(tokens) + [tokens[j] + "_" + tokens[j + 1] for j in range(len(tokens) - 1)]
            for g in grams:
                h = hashlib.md5(g.encode("utf-8")).digest()
                idx = int.from_bytes(h[:4], "little") % self.dim
                sign = 1.0 if (h[4] % 2 == 0) else -1.0
                vecs[i, idx] += sign
            norm = np.linalg.norm(vecs[i])
            if norm > 0:
                vecs[i] /= norm
        return vecs


class SentenceTransformerEmbedder:
    name = _MODEL_NAME

    def __init__(self, model_name: str = _MODEL_NAME):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts):
        return np.asarray(
            self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )


def get_embedder():
    force_fallback = os.environ.get("USE_HASHING_EMBEDDER", "0") == "1"
    if not force_fallback:
        try:
            return SentenceTransformerEmbedder()
        except Exception as e:
            print(f"[embeddings] sentence-transformers unavailable ({e}); using hashing fallback.")
    return HashingEmbedder()
