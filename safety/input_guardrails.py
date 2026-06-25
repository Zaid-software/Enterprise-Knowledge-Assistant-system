import re
from typing import Tuple, Optional

from agents.schemas import IncidentLogEntry, GuardrailDecision

# ---------- 1. Prompt injection detection ----------

INJECTION_PATTERNS = [
    r"ignore (all )?(previous|prior|above) instructions",
    r"disregard (all )?(previous|prior|above) instructions",
    r"reveal (your |the )?system prompt",
    r"show me (your |the )?system prompt",
    r"you are now (a |an )?(?!intern|manager|admin)\w+",  # role-swap attempt
    r"act as (if you (are|were)|a)\b",
    r"pretend (you are|to be)",
    r"forget (everything|all) (you('ve| have)? (been told|learned))",
    r"new instructions:?",
    r"do anything now",
    r"\bdan\b mode",
    r"override (your )?(safety|guardrails?|rules)",
    r"this is (a |an )?(test|admin|developer) (mode|override)",
    r"print (your |the )?(instructions|prompt|rules)",
]
_INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)


def detect_prompt_injection(text: str) -> Optional[str]:
    """Returns the matched pattern description if injection detected, else None."""
    m = _INJECTION_RE.search(text)
    if m:
        return f"injection_pattern_match: '{m.group(0)}'"
    return None


# ---------- 2. PII filter ----------

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_RE = re.compile(r"\b(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
CREDIT_CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,16}\b")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


def detect_and_redact_pii(text: str) -> Tuple[str, list]:
    """Returns (redacted_text, list_of_pii_types_found)."""
    found = []
    redacted = text

    if SSN_RE.search(redacted):
        found.append("ssn")
        redacted = SSN_RE.sub("[REDACTED_SSN]", redacted)
    if CREDIT_CARD_RE.search(redacted):
        # avoid false-positiving on long phone numbers already caught; check digit count loosely
        for match in CREDIT_CARD_RE.finditer(redacted):
            digits = re.sub(r"\D", "", match.group(0))
            if len(digits) in (13, 14, 15, 16):
                found.append("credit_card")
                redacted = redacted.replace(match.group(0), "[REDACTED_CC]")
    if EMAIL_RE.search(redacted):
        found.append("email")
        redacted = EMAIL_RE.sub("[REDACTED_EMAIL]", redacted)
    if PHONE_RE.search(redacted):
        found.append("phone")
        redacted = PHONE_RE.sub("[REDACTED_PHONE]", redacted)

    return redacted, found


# ---------- 3. Topic / policy filter ----------

OFF_SCOPE_PATTERNS = [
    r"\bdiagnos(e|is|ing)\b.*\b(me|my)\b",
    r"what (medication|drug|dosage)",
    r"should i sue",
    r"is this legal(ly)? (binding|enforceable)",
    r"give me legal advice",
    r"what'?s (my|the) (ceo|cfo|coworker'?s|manager'?s) salary",
    r"compensation (band|details) for (?!my own)",
    r"social security number of",
]
_OFF_SCOPE_RE = re.compile("|".join(OFF_SCOPE_PATTERNS), re.IGNORECASE)


def detect_off_scope(text: str) -> Optional[str]:
    m = _OFF_SCOPE_RE.search(text)
    if m:
        return f"off_scope_pattern_match: '{m.group(0)}'"
    return None


# ---------- Orchestrating function ----------

def run_input_guardrails(user_message: str) -> Tuple[GuardrailDecision, str, list]:
    """
    Returns (decision, sanitized_or_original_message, incident_log_entries)
    decision: REJECT (injection or off-scope) | REDACT (PII found, message can proceed redacted) | PASS
    """
    incidents = []

    injection_match = detect_prompt_injection(user_message)
    if injection_match:
        incidents.append(IncidentLogEntry(
            rule_triggered=injection_match,
            redacted_input=user_message[:120],
            decision=GuardrailDecision.REJECT.value,
            stage="input",
        ))
        return GuardrailDecision.REJECT, "", incidents

    off_scope_match = detect_off_scope(user_message)
    if off_scope_match:
        incidents.append(IncidentLogEntry(
            rule_triggered=off_scope_match,
            redacted_input=user_message[:120],
            decision=GuardrailDecision.REJECT.value,
            stage="input",
        ))
        return GuardrailDecision.REJECT, "", incidents

    redacted_message, pii_found = detect_and_redact_pii(user_message)
    if pii_found:
        incidents.append(IncidentLogEntry(
            rule_triggered=f"pii_detected: {','.join(pii_found)}",
            redacted_input=redacted_message[:120],
            decision=GuardrailDecision.REDACT.value,
            stage="input",
        ))
        return GuardrailDecision.REDACT, redacted_message, incidents

    return GuardrailDecision.PASS, user_message, incidents
