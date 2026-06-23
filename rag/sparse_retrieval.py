import math
import os
import re
from collections import Counter

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str):
    return _WORD_RE.findall(text.lower())


class SimpleBM25:

    def __init__(self, corpus_tokens, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus_tokens = corpus_tokens
        self.N = len(corpus_tokens)
        self.doc_lens = [len(d) for d in corpus_tokens]
        self.avgdl = sum(self.doc_lens) / self.N if self.N else 0
        self.df = Counter()
        for doc in corpus_tokens:
            for term in set(doc):
                self.df[term] += 1
        self.idf = {
            term: math.log(1 + (self.N - df + 0.5) / (df + 0.5))
            for term, df in self.df.items()
        }
        self.doc_term_counts = [Counter(d) for d in corpus_tokens]

    def get_scores(self, query_tokens):
        scores = [0.0] * self.N
        for term in query_tokens:
            if term not in self.idf:
                continue
            idf = self.idf[term]
            for i in range(self.N):
                f = self.doc_term_counts[i].get(term, 0)
                if f == 0:
                    continue
                dl = self.doc_lens[i]
                denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[i] += idf * (f * (self.k1 + 1)) / denom
        return scores


class BM25Index:
    def __init__(self, chunk_ids, texts):
        self.chunk_ids = chunk_ids
        self.use_rank_bm25 = False
        tokenized = [_tokenize(t) for t in texts]
        if os.environ.get("USE_SIMPLE_BM25", "0") != "1":
            try:
                from rank_bm25 import BM25Okapi
                self.bm25 = BM25Okapi(tokenized)
                self.use_rank_bm25 = True
            except Exception as e:
                print(f"[bm25] rank_bm25 unavailable ({e}); using built-in BM25.")
                self.bm25 = SimpleBM25(tokenized)
        else:
            self.bm25 = SimpleBM25(tokenized)

    def search(self, query: str, top_k: int):
        q_tokens = _tokenize(query)
        scores = self.bm25.get_scores(q_tokens)
        ranked = sorted(zip(self.chunk_ids, scores), key=lambda x: -x[1])
        return ranked[:top_k]
