import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional, Dict, Any

try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    from agents._pydantic_shim import BaseModel, Field
    PYDANTIC_AVAILABLE = False


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MessageType(str, Enum):
    USER_REQUEST = "USER_REQUEST"
    RETRIEVAL_REQUEST = "RETRIEVAL_REQUEST"
    RETRIEVAL_RESULT = "RETRIEVAL_RESULT"
    SYNTHESIS_REQUEST = "SYNTHESIS_REQUEST"
    SYNTHESIS_RESULT = "SYNTHESIS_RESULT"
    SAFETY_REVIEW_REQUEST = "SAFETY_REVIEW_REQUEST"
    SAFETY_VERDICT = "SAFETY_VERDICT"
    FINAL_ANSWER = "FINAL_ANSWER"
    INCIDENT_LOG = "INCIDENT_LOG"


class GuardrailDecision(str, Enum):
    PASS = "pass"
    REDACT = "redact"
    REJECT = "reject"
    REGENERATE = "regenerate"
    APPROVE = "approve"


# ---------- Payloads ----------

class RetrievalRequest(BaseModel):
    query: str
    top_k: int = 5
    user_role: str = "intern"
    correlation_id: str = Field(default_factory=new_id)


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    text: str
    score: float
    fused_score: float = 0.0


class RetrievalResult(BaseModel):
    query: str
    chunks: List[RetrievedChunk]
    rbac_blocked_chunk_ids: List[str] = Field(default_factory=list)
    bm25_top_score: float = 0.0
    retrieval_trace: Dict[str, Any] = Field(default_factory=dict)
    correlation_id: str = Field(default_factory=new_id)


class SynthesisRequest(BaseModel):
    question: str
    chunks: List[RetrievedChunk]
    critique: Optional[str] = None  # populated on regeneration feedback loop
    round_number: int = 1
    correlation_id: str = Field(default_factory=new_id)


class SynthesisResult(BaseModel):
    draft_answer: str
    cited_chunk_ids: List[str] = Field(default_factory=list)
    correlation_id: str = Field(default_factory=new_id)


class SafetyReviewRequest(BaseModel):
    question: str
    draft_answer: str
    cited_chunk_ids: List[str]
    available_chunks: List[RetrievedChunk]
    correlation_id: str = Field(default_factory=new_id)


class SafetyVerdict(BaseModel):
    decision: GuardrailDecision
    reason: str
    redacted_answer: Optional[str] = None
    unsupported_claims: List[str] = Field(default_factory=list)
    correlation_id: str = Field(default_factory=new_id)


class IncidentLogEntry(BaseModel):
    timestamp: str = Field(default_factory=now_iso)
    rule_triggered: str
    redacted_input: str
    decision: str
    stage: str  # "input" or "output"
    correlation_id: str = Field(default_factory=new_id)


class FinalAnswer(BaseModel):
    answer: str
    citations: List[str]
    safety_decision: GuardrailDecision
    rounds_used: int
    correlation_id: str = Field(default_factory=new_id)


# ---------- Envelope (A2A pattern) ----------

class Envelope(BaseModel):
    envelope_id: str = Field(default_factory=new_id)
    sender: str
    recipient: str
    correlation_id: str
    message_type: MessageType
    timestamp: str = Field(default_factory=now_iso)
    payload: Dict[str, Any] = Field(default_factory=dict)

    @staticmethod
    def wrap(sender: str, recipient: str, correlation_id: str,
             message_type: MessageType, payload_model) -> "Envelope":
        payload_dict = payload_model.model_dump() if hasattr(payload_model, "model_dump") else dict(payload_model)
        return Envelope(
            sender=sender,
            recipient=recipient,
            correlation_id=correlation_id,
            message_type=message_type,
            payload=payload_dict,
        )
