from typing import Dict, List, Tuple

RRF_K = 60  # standard constant for web-scale corpora (Cormack et al., 2009);
            # see reciprocal_rank_fusion() for why we override this for small corpora
DENSE_WEIGHT = 1.0
SPARSE_WEIGHT = 1.3


def _confidence_boost(results: List[Tuple[str, float]], base_score_range: float) -> float:
    if len(results) < 2 or base_score_range <= 0:
        return 1.0
    gap = results[0][1] - results[1][1]
    normalized_gap = gap / base_score_range
    # cap the boost so one query can't completely dominate fusion either
    return 1.0 + min(1.5, max(0.0, normalized_gap))


def reciprocal_rank_fusion(
    dense_results: List[Tuple[str, float]],
    sparse_results: List[Tuple[str, float]],
    k: int = RRF_K,
    dense_weight: float = DENSE_WEIGHT,
    sparse_weight: float = SPARSE_WEIGHT,
    dominance_ratio: float = 2.0,
) -> List[Tuple[str, float, dict]]:
 
    dense_rank = {cid: i for i, (cid, _) in enumerate(dense_results)}
    sparse_rank = {cid: i for i, (cid, _) in enumerate(sparse_results)}
    dense_score = {cid: s for cid, s in dense_results}
    sparse_score = {cid: s for cid, s in sparse_results}

    n_candidates = max(len(dense_rank), len(sparse_rank), 1)
    effective_k = max(5, min(k, n_candidates))

    sparse_range = sparse_results[0][1] if sparse_results else 0.0
    dense_range = 1.0

    sparse_boost = _confidence_boost(sparse_results, sparse_range)
    dense_boost = _confidence_boost(dense_results, dense_range)

    all_ids = set(dense_rank) | set(sparse_rank)
    fused = []
    for cid in all_ids:
        rrf = 0.0
        if cid in dense_rank:
            rrf += (dense_weight * dense_boost) / (effective_k + dense_rank[cid] + 1)
        if cid in sparse_rank:
            rrf += (sparse_weight * sparse_boost) / (effective_k + sparse_rank[cid] + 1)
        fused.append((cid, rrf, {
            "dense_rank": dense_rank.get(cid),
            "dense_score": dense_score.get(cid),
            "sparse_rank": sparse_rank.get(cid),
            "sparse_score": sparse_score.get(cid),
        }))

    fused.sort(key=lambda x: -x[1])

    # Dominance override
    if len(sparse_results) >= 2 and sparse_results[1][1] > 0:
        ratio = sparse_results[0][1] / sparse_results[1][1]
        if ratio >= dominance_ratio:
            dominant_id = sparse_results[0][0]
            fused = [f for f in fused if f[0] != dominant_id]
            dbg = {
                "dense_rank": dense_rank.get(dominant_id),
                "dense_score": dense_score.get(dominant_id),
                "sparse_rank": sparse_rank.get(dominant_id),
                "sparse_score": sparse_score.get(dominant_id),
                "dominance_override": True,
                "dominance_ratio": round(ratio, 2),
            }
            # score it just above the current top so ordering stays monotonic for logging
            top_score = fused[0][1] if fused else 1.0
            fused.insert(0, (dominant_id, top_score + 0.01, dbg))

    return fused
