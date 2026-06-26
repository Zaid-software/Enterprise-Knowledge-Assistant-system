from agents.schemas import SafetyReviewRequest, SafetyVerdict, GuardrailDecision, IncidentLogEntry
from agents.llm_client import get_reviewer_llm
from safety.output_guardrails import (
    run_output_guardrails, build_safety_reviewer_prompt, parse_reviewer_verdict,
)


def run_safety_reviewer(req: SafetyReviewRequest, user_role: str = "intern"):
    # Layer 1: deterministic checks
    det_verdict, incidents = run_output_guardrails(req, user_role=user_role)

    layer_breakdown = {
        "deterministic_layer": {
            "decision": det_verdict.decision.value,
            "reason": det_verdict.reason,
        }
    }

    if det_verdict.decision != GuardrailDecision.APPROVE:
        # fail fast: don't even spend an LLM call if deterministic checks already failed
        layer_breakdown["llm_layer"] = {"decision": "skipped", "reason": "deterministic layer already blocked"}
        return det_verdict, incidents, layer_breakdown

    # Layer 2: dual-LLM semantic review (isolated, tool-free)
    reviewer = get_reviewer_llm()
    prompt = build_safety_reviewer_prompt(req)
    raw_output = reviewer.review(prompt)
    llm_verdict = parse_reviewer_verdict(raw_output)

    layer_breakdown["llm_layer"] = {
        "model": getattr(reviewer, "name", "unknown"),
        "decision": llm_verdict.decision.value,
        "reason": llm_verdict.reason,
        "raw_output": raw_output,
    }

    if llm_verdict.decision != GuardrailDecision.APPROVE:
        incidents.append(IncidentLogEntry(
            rule_triggered=f"dual_llm_reviewer_flag: {llm_verdict.reason}",
            redacted_input=req.draft_answer[:120],
            decision=llm_verdict.decision.value,
            stage="output",
        ))

    return llm_verdict, incidents, layer_breakdown
