import json
import os
import numpy as np


class NumpyFlatIndex:

    backend = "numpy-flat-fallback"

    def __init__(self, dim: int):
        self.dim = dim
        self.vectors = np.zeros((0, dim), dtype=np.float32)
        self.ids = []  # chunk_ids, parallel to rows

    def add(self, vectors: np.ndarray, ids):
        self.vectors = np.vstack([self.vectors, vectors]) if self.vectors.shape[0] else vectors
        self.ids.extend(ids)

    def search(self, query_vec: np.ndarray, top_k: int):
        if self.vectors.shape[0] == 0:
            return []
        # vectors are already L2-normalized -> dot product == cosine similarity
        scores = self.vectors @ query_vec
        top_k = min(top_k, len(scores))
        top_idx = np.argpartition(-scores, top_k - 1)[:top_k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(self.ids[i], float(scores[i])) for i in top_idx]

    def save(self, path: str):
        np.savez(path, vectors=self.vectors, ids=np.array(self.ids, dtype=object))

    @classmethod
    def load(cls, path: str):
        data = np.load(path, allow_pickle=True)
        idx = cls(dim=data["vectors"].shape[1])
        idx.vectors = data["vectors"]
        idx.ids = list(data["ids"])
        return idx


class FaissFlatIndex:
    backend = "faiss"

    def __init__(self, dim: int):
        import faiss
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # inner product on normalized vecs == cosine
        self.ids = []

    def add(self, vectors: np.ndarray, ids):
        self.index.add(vectors)
        self.ids.extend(ids)

    def search(self, query_vec: np.ndarray, top_k: int):
        if self.index.ntotal == 0:
            return []
        scores, idxs = self.index.search(query_vec.reshape(1, -1), min(top_k, self.index.ntotal))
        return [(self.ids[i], float(s)) for s, i in zip(scores[0], idxs[0]) if i != -1]

    def save(self, path: str):
        import faiss
        faiss.write_index(self.index, path + ".faiss")
        with open(path + ".ids.json", "w") as f:
            json.dump(self.ids, f)

    @classmethod
    def load(cls, path: str):
        import faiss
        index = faiss.read_index(path + ".faiss")
        obj = cls.__new__(cls)
        obj.index = index
        obj.dim = index.d
        with open(path + ".ids.json") as f:
            obj.ids = json.load(f)
        return obj


def get_vector_store(dim: int):
    force_fallback = os.environ.get("USE_NUMPY_VECTORSTORE", "0") == "1"
    if not force_fallback:
        try:
            return FaissFlatIndex(dim)
        except Exception as e:
            print(f"[vectorstore] FAISS unavailable ({e}); using numpy fallback.")
    return NumpyFlatIndex(dim)
