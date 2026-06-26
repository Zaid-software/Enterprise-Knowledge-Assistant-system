import json
import os
import numpy as np

from rag.embeddings import get_embedder
from rag.vector_store import get_vector_store, FaissFlatIndex, NumpyFlatIndex
from rag.sparse_retrieval import BM25Index
from rag.fusion import reciprocal_rank_fusion
from rag.reranking import mmr_rerank, get_cross_encoder

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORE_DIR = os.path.join(BASE_DIR, "rag", "store")

ROLE_RANK = {"intern": 0, "manager": 1, "admin": 2}


class Retriever:
    def __init__(self):
        with open(os.path.join(STORE_DIR, "chunk_meta.json"), encoding="utf-8") as f:
            self.chunk_meta = json.load(f)
        with open(os.path.join(STORE_DIR, "ingest_config.json")) as f:
            self.config = json.load(f)

        self.embedder = get_embedder()
        self.chunk_ids = list(self.chunk_meta.keys())
        self.texts = [self.chunk_meta[cid]["text"] for cid in self.chunk_ids]

        # Load (or rebuild) the vector index
        vec_path = os.path.join(STORE_DIR, "vector_index")
        self.vector_store = self._load_vector_store(vec_path)

        # Cache all chunk vectors in memory too (needed for MMR similarity matrix)
        self._chunk_vectors = self.embedder.embed(self.texts)

        self.bm25 = BM25Index(self.chunk_ids, self.texts)
        self.cross_encoder = get_cross_encoder()

    def _load_vector_store(self, path):
        try:
            return FaissFlatIndex.load(path)
        except Exception:
            try:
                return NumpyFlatIndex.load(path + ".npz")
            except Exception as e:
                raise RuntimeError(
                    "Could not load vector store. Run `python -m rag.ingest` first."
                ) from e

    def _vec_for_chunk(self, chunk_id: str):
        idx = self.chunk_ids.index(chunk_id)
        return self._chunk_vectors[idx]

    def retrieve(self, query: str, top_k: int = 5, candidate_pool: int = 25, user_role: str = "intern"):
        # 1. dense -- retrieve a generous pool since a noisy embedder can rank
        #    a truly relevant chunk outside a narrow top-N
        dense_results = self.vector_store.search(q_vec := self.embedder.embed([query])[0], min(candidate_pool, len(self.chunk_ids)))

        # 2. sparse -- same generous pool, so a strong keyword match isn't
        #    dropped before fusion even runs
        sparse_results = self.bm25.search(query, min(candidate_pool, len(self.chunk_ids)))

        # 3. RRF fusion over the union of both candidate pools
        fused = reciprocal_rank_fusion(dense_results, sparse_results)
        fused_top = fused[:candidate_pool]

        # 4. RBAC filter -- drop chunks the user's role can't see, BEFORE rerank/return
        user_rank = ROLE_RANK.get(user_role, 0)
        visible = [
            (cid, score, dbg) for cid, score, dbg in fused_top
            if ROLE_RANK.get(self.chunk_meta[cid]["min_role"], 0) <= user_rank
        ]
        blocked = [
            (cid, score, dbg) for cid, score, dbg in fused_top
            if ROLE_RANK.get(self.chunk_meta[cid]["min_role"], 0) > user_rank
        ]

        before_rerank = [
            {
                "chunk_id": cid,
                "fused_score": round(score, 5),
                "dense_rank": dbg["dense_rank"],
                "dense_score": round(dbg["dense_score"], 5) if dbg["dense_score"] is not None else None,
                "sparse_rank": dbg["sparse_rank"],
                "sparse_score": round(dbg["sparse_score"], 5) if dbg["sparse_score"] is not None else None,
            }
            for cid, score, dbg in visible
        ]

        # 5. rerank: cross-encoder (if available) re-scores relevance, MMR adds diversity
        candidate_ids = [cid for cid, _, _ in visible]
        candidate_scores = [score for _, score, _ in visible]

        if self.cross_encoder is not None and candidate_ids:
            ce_candidates = [(cid, self.chunk_meta[cid]["text"]) for cid in candidate_ids]
            ce_ranked = self.cross_encoder.rerank(query, ce_candidates, top_k=len(candidate_ids))
            ce_score_map = {cid: s for cid, s in ce_ranked}
            candidate_scores = [float(ce_score_map.get(cid, 0.0)) for cid in candidate_ids]

        candidate_vectors = np.array([self._vec_for_chunk(cid) for cid in candidate_ids]) if candidate_ids else np.zeros((0, self.embedder.dim))
        reranked = mmr_rerank(candidate_ids, candidate_vectors, candidate_scores, top_k=top_k)

        after_rerank = [
            {"chunk_id": cid, "rerank_score": round(score, 5)}
            for cid, score in reranked
        ]

        results = []
        fused_score_by_id = {cid: score for cid, score, _ in visible}
        # bm25_top_score must reflect what the user can actually see post-RBAC,
        # not the raw pre-filter pool -- otherwise a high-confidence match the
        # user is blocked from seeing would wrongly pass the relevance gate
        # and let the synthesizer answer from irrelevant leftover chunks.
        visible_ids = {cid for cid, _, _ in visible}
        visible_sparse_scores = [s for cid, s in sparse_results if cid in visible_ids]
        bm25_top_score = visible_sparse_scores[0] if visible_sparse_scores else 0.0
        for cid, score in reranked:
            meta = self.chunk_meta[cid]
            results.append({
                "chunk_id": cid,
                "doc_id": meta["doc_id"],
                "title": meta["title"],
                "text": meta["text"],
                "score": round(score, 5),
                "fused_score": round(fused_score_by_id.get(cid, 0.0), 5),
                "bm25_top_score": round(bm25_top_score, 5),
            })

        return {
            "query": query,
            "user_role": user_role,
            "results": results,
            "rbac_blocked_chunk_ids": [cid for cid, _, _ in blocked],
            "bm25_top_score": round(bm25_top_score, 5),
            "trace": {
                "dense_candidates": [{"chunk_id": cid, "score": round(s, 5)} for cid, s in dense_results],
                "sparse_candidates": [{"chunk_id": cid, "score": round(s, 5)} for cid, s in sparse_results],
                "fusion_strategy": "reciprocal_rank_fusion",
                "before_rerank": before_rerank,
                "rerank_method": "cross_encoder+mmr" if self.cross_encoder else "mmr_only",
                "after_rerank": after_rerank,
            },
        }
