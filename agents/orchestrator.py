import os
from typing import List

from agents.schemas import (
    Envelope, MessageType, GuardrailDecision,
    RetrievalRequest, SynthesisRequest, SafetyReviewRequest,
    FinalAnswer, IncidentLogEntry,
)
from agents.retriever_agent import run_retriever_agent
from agents.synthesizer_agent import run_synthesizer
from agents.safety_reviewer_agent import run_safety_reviewer
from safety.input_guardrails import run_input_guardrails

MAX_ROUNDS = 3
MIN_RELEVANCE_SCORE = float(os.environ.get("MIN_RELEVANCE_SCORE", "5.0"))


class Orchestrator:
    def __init__(self):
        self.trace: List[dict] = []
        self.incidents: List[dict] = []

    def _log(self, envelope: Envelope, summary: str = ""):
        entry = envelope.model_dump()
        entry["_summary"] = summary
        self.trace.append(entry)

    def _log_incidents(self, incidents: List[IncidentLogEntry]):
        for inc in incidents:
            self.incidents.append(inc.model_dump())

    def handle_request(self, user_message: str, user_role: str = "intern", top_k: int = 5) -> dict:
        correlation_id = None

        # ---- Step 1: input guardrails ----
        decision, sanitized_message, incidents = run_input_guardrails(user_message)
        self._log_incidents(incidents)

        if decision == GuardrailDecision.REJECT:
            env = Envelope(
                sender="orchestrator", recipient="user",
                correlation_id="rejected", message_type=MessageType.FINAL_ANSWER,
                payload={"rejected": True},
            )
            self._log(env, summary="Input guardrail REJECTED the request before any agent ran.")
            return {
                "final_answer": "I can't help with that request — it was flagged by an input safety check.",
                "citations": [],
                "safety_decision": "reject",
                "rounds_used": 0,
                "trace": self.trace,
                "incidents": self.incidents,
            }

        working_message = sanitized_message  # redacted if PII was found, else original

        # ---- Step 2: retrieval ----
        retrieval_req = RetrievalRequest(query=working_message, top_k=top_k, user_role=user_role)
        correlation_id = retrieval_req.correlation_id

        self._log(
            Envelope.wrap("orchestrator", "retriever_agent", correlation_id,
                          MessageType.RETRIEVAL_REQUEST, retrieval_req),
            summary=f"orchestrator -> retriever_agent: query='{working_message[:60]}', top_k={top_k}, role={user_role}",
        )
        retrieval_result = run_retriever_agent(retrieval_req)
        self._log(
            Envelope.wrap("retriever_agent", "orchestrator", correlation_id,
                          MessageType.RETRIEVAL_RESULT, retrieval_result),
            summary=f"retriever_agent -> orchestrator: {len(retrieval_result.chunks)} chunks "
                    f"(top: {retrieval_result.chunks[0].chunk_id if retrieval_result.chunks else 'none'})",
        )

        if not retrieval_result.chunks:
            final = "I don't have enough information on this topic in the available documents."
            self._log(
                Envelope(sender="orchestrator", recipient="user", correlation_id=correlation_id,
                         message_type=MessageType.FINAL_ANSWER, payload={"answer": final}),
                summary="No chunks retrieved; returning explicit 'I don't know'.",
            )
            return {
                "final_answer": final, "citations": [], "safety_decision": "approve",
                "rounds_used": 0, "trace": self.trace, "incidents": self.incidents,
            }

        top_score = retrieval_result.bm25_top_score
        if top_score < MIN_RELEVANCE_SCORE:
            final = "I don't have enough information on this topic in the available documents."
            self._log(
                Envelope(sender="orchestrator", recipient="user", correlation_id=correlation_id,
                         message_type=MessageType.FINAL_ANSWER, payload={"answer": final, "abstained": True}),
                summary=f"Relevance gate: top score {top_score:.4f} < threshold {MIN_RELEVANCE_SCORE}; "
                        f"abstaining instead of calling synthesizer.",
            )
            return {
                "final_answer": final, "citations": [], "safety_decision": "approve",
                "rounds_used": 0, "trace": self.trace, "incidents": self.incidents,
            }

        # ---- Step 3-5: synthesis <-> safety review feedback loop ----
        critique = None
        round_number = 1
        final_answer_text = None
        final_citations = []
        final_decision = GuardrailDecision.REGENERATE

        while round_number <= MAX_ROUNDS:
            synthesis_req = SynthesisRequest(
                question=working_message, chunks=retrieval_result.chunks,
                critique=critique, round_number=round_number, correlation_id=correlation_id,
            )
            self._log(
                Envelope.wrap("orchestrator", "synthesizer_agent", correlation_id,
                              MessageType.SYNTHESIS_REQUEST, synthesis_req),
                summary=f"orchestrator -> synthesizer_agent: round {round_number}"
                        + (f", critique='{critique}'" if critique else ""),
            )
            synthesis_result = run_synthesizer(synthesis_req)
            self._log(
                Envelope.wrap("synthesizer_agent", "orchestrator", correlation_id,
                              MessageType.SYNTHESIS_RESULT, synthesis_result),
                summary=f"synthesizer_agent -> orchestrator: draft ({len(synthesis_result.draft_answer)} chars), "
                        f"cites={synthesis_result.cited_chunk_ids}",
            )

            safety_req = SafetyReviewRequest(
                question=working_message,
                draft_answer=synthesis_result.draft_answer,
                cited_chunk_ids=synthesis_result.cited_chunk_ids,
                available_chunks=retrieval_result.chunks,
                correlation_id=correlation_id,
            )
            self._log(
                Envelope.wrap("orchestrator", "safety_reviewer_agent", correlation_id,
                              MessageType.SAFETY_REVIEW_REQUEST, safety_req),
                summary="orchestrator -> safety_reviewer_agent: requesting verdict (dual-LLM, isolated, no tool access)",
            )
            verdict, sr_incidents, layer_breakdown = run_safety_reviewer(safety_req, user_role=user_role)
            self._log_incidents(sr_incidents)
            self._log(
                Envelope.wrap("safety_reviewer_agent", "orchestrator", correlation_id,
                              MessageType.SAFETY_VERDICT, verdict),
                summary=f"safety_reviewer_agent -> orchestrator: {verdict.decision.value} "
                        f"(deterministic={layer_breakdown['deterministic_layer']['decision']}, "
                        f"llm={layer_breakdown.get('llm_layer', {}).get('decision')})",
            )

            if verdict.decision == GuardrailDecision.APPROVE:
                final_answer_text = synthesis_result.draft_answer
                final_citations = synthesis_result.cited_chunk_ids
                final_decision = GuardrailDecision.APPROVE
                break

            if verdict.decision == GuardrailDecision.REDACT:
                final_answer_text = verdict.redacted_answer or synthesis_result.draft_answer
                final_citations = synthesis_result.cited_chunk_ids
                final_decision = GuardrailDecision.REDACT
                break

            # REGENERATE: feedback loop -- re-dispatch to synthesizer with critique
            critique = verdict.reason
            round_number += 1
            final_decision = GuardrailDecision.REGENERATE
            final_answer_text = None
            final_citations = []

        if final_decision == GuardrailDecision.REGENERATE and round_number > MAX_ROUNDS:
            final_answer_text = (
                "I wasn't able to produce a safely verified answer to this question "
                f"after {MAX_ROUNDS} attempts, so I'm not going to guess. This can happen "
                "when source documents conflict, are insufficient, or (rarely) contain "
                "suspicious embedded content. Please rephrase the question or contact "
                "a human admin if this persists."
            )
            final_citations = []

        final = FinalAnswer(
            answer=final_answer_text,
            citations=final_citations,
            safety_decision=final_decision,
            rounds_used=round_number if round_number <= MAX_ROUNDS else MAX_ROUNDS,
            correlation_id=correlation_id,
        )
        self._log(
            Envelope.wrap("orchestrator", "user", correlation_id, MessageType.FINAL_ANSWER, final),
            summary=f"orchestrator -> user: final answer delivered (decision={final_decision.value}, "
                    f"rounds={final.rounds_used})",
        )

        return {
            "final_answer": final.answer,
            "citations": final.citations,
            "safety_decision": final_decision.value,
            "rounds_used": final.rounds_used,
            "trace": self.trace,
            "incidents": self.incidents,
        }
