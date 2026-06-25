import os
import re
import requests

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SYNTHESIZER_MODEL = os.environ.get("SYNTHESIZER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
REVIEWER_MODEL = os.environ.get("REVIEWER_MODEL", "mistralai/mistral-7b-instruct:free")


def _call_openrouter(model: str, system_prompt: str, user_prompt: str, max_tokens: int = 500) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }
    resp = requests.post(OPENROUTER_URL, headers=headers, json=body, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


class StubSynthesizerLLM:

    name = "stub-extractive-synthesizer"

    def synthesize(self, question: str, chunks):
        if not chunks:
            return "I don't have enough information on this topic in the available documents.", []

        q_words = set(re.findall(r"[a-z0-9]+", question.lower()))
        STOPWORDS = {"the", "a", "an", "is", "are", "was", "were", "what", "when", "where",
                     "who", "how", "did", "does", "do", "of", "in", "on", "to", "for", "and"}
        q_words -= STOPWORDS

        best = chunks[0]
        best_text = best.text if hasattr(best, "text") else best["text"]
        sentences = re.split(r"(?<=[.!?])\s+", best_text)
        scored = []
        for s in sentences:
            s_words = set(re.findall(r"[a-z0-9]+", s.lower()))
            overlap = len(s_words & q_words)
            scored.append((overlap, s))
        scored.sort(key=lambda x: -x[0])

        if scored[0][0] == 0:
            return "I don't have enough information on this topic in the available documents.", []

        top_sentences = [s for overlap, s in scored[:2] if overlap > 0]

        chunk_id = best.chunk_id if hasattr(best, "chunk_id") else best["chunk_id"]
        answer = " ".join(top_sentences) + f" [Source: {chunk_id}]"
        return answer, [chunk_id]


class StubReviewerLLM:

    name = "stub-rule-based-reviewer"

    def review(self, prompt: str) -> str:
        # The deterministic checks already ran upstream in run_output_guardrails;
        # this stub's job is just to emulate a *second*, semantic pass that
        # looks for injected instructions inside the chunk text shown in the prompt.
        injected_instruction_signals = [
            "ignore previous instructions", "ignore all previous", "reveal the system prompt",
            "disregard", "new instructions", "you are now", "act as if",
        ]
        lowered = prompt.lower()
        for sig in injected_instruction_signals:
            if sig in lowered:
                return f"VERDICT: regenerate | REASON: possible indirect prompt injection detected in retrieved content ('{sig}')"
        return "VERDICT: approve | REASON: no additional issues found by semantic review"


def get_synthesizer_llm():
    if OPENROUTER_API_KEY:
        return RealLLMSynthesizer()
    return StubSynthesizerLLM()


def get_reviewer_llm():
    if OPENROUTER_API_KEY:
        return RealLLMReviewer()
    return StubReviewerLLM()


class RealLLMSynthesizer:
    name = SYNTHESIZER_MODEL

    def synthesize_raw(self, system_prompt: str, user_prompt: str) -> str:
        return _call_openrouter(SYNTHESIZER_MODEL, system_prompt, user_prompt, max_tokens=600)


class RealLLMReviewer:
    name = REVIEWER_MODEL

    def review(self, prompt: str) -> str:
        from safety.output_guardrails import SAFETY_REVIEWER_SYSTEM_PROMPT
        return _call_openrouter(REVIEWER_MODEL, SAFETY_REVIEWER_SYSTEM_PROMPT, prompt, max_tokens=150)
