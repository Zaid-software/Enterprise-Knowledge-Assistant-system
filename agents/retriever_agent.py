from agents.schemas import RetrievalRequest, RetrievalResult, RetrievedChunk

_retriever_singleton = None


def _get_retriever():
    global _retriever_singleton
    if _retriever_singleton is None:
        from rag.retriever import Retriever
        _retriever_singleton = Retriever()
    return _retriever_singleton


def run_retriever_agent(req: RetrievalRequest) -> RetrievalResult:
    retriever = _get_retriever()
    raw = retriever.retrieve(req.query, top_k=req.top_k, user_role=req.user_role)

    chunks = [
        RetrievedChunk(chunk_id=r["chunk_id"], doc_id=r["doc_id"], title=r["title"],
                        text=r["text"], score=r["score"], fused_score=r.get("fused_score", 0.0))
        for r in raw["results"]
    ]
    return RetrievalResult(
        query=req.query,
        chunks=chunks,
        rbac_blocked_chunk_ids=raw["rbac_blocked_chunk_ids"],
        bm25_top_score=raw.get("bm25_top_score", 0.0),
        retrieval_trace=raw["trace"],
        correlation_id=req.correlation_id,
    )
