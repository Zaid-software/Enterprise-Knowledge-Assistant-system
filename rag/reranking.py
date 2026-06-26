import os
from typing import List, Tuple
import numpy as np


def mmr_rerank(
    candidate_ids: List[str],
    candidate_vectors: np.ndarray,
    relevance_scores: List[float],
    top_k: int,
    lambda_param: float = 0.7,
) -> List[Tuple[str, float]]:

    if len(candidate_ids) == 0:
        return []
    n = len(candidate_ids)
    top_k = min(top_k, n)

    relevance = np.array(relevance_scores, dtype=np.float32)
    if relevance.max() > 0:
        relevance = relevance / relevance.max()

    selected = []
    selected_idx = []
    remaining = list(range(n))

    # similarity matrix between all candidates (cosine, vectors are normalized)
    sim_matrix = candidate_vectors @ candidate_vectors.T

    while len(selected) < top_k and remaining:
        best_score = -1e9
        best_i = None
        for i in remaining:
            if selected_idx:
                max_sim_to_selected = max(sim_matrix[i, j] for j in selected_idx)
            else:
                max_sim_to_selected = 0.0
            mmr_score = lambda_param * relevance[i] - (1 - lambda_param) * max_sim_to_selected
            if mmr_score > best_score:
                best_score = mmr_score
                best_i = i
        selected.append((candidate_ids[best_i], float(best_score)))
        selected_idx.append(best_i)
        remaining.remove(best_i)

    return selected


class CrossEncoderReranker:
    name = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(self.name)

    def rerank(self, query: str, candidates: List[Tuple[str, str]], top_k: int):
        """candidates: list of (chunk_id, chunk_text)."""
        pairs = [[query, text] for _, text in candidates]
        scores = self.model.predict(pairs)
        ranked = sorted(zip([c[0] for c in candidates], scores), key=lambda x: -x[1])
        return ranked[:top_k]


def get_cross_encoder():
    if os.environ.get("DISABLE_CROSS_ENCODER", "0") == "1":
        return None
    try:
        return CrossEncoderReranker()
    except Exception as e:
        print(f"[rerank] Cross-encoder unavailable ({e}); using MMR only.")
        return None
