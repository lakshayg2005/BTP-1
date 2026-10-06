"""Explainer v1: deterministic, evidence-only explanation (no LLM).

It is the fallback for the LLM explainer and the reference for its faithfulness check:
every sentence here comes from an Evidence.description, so nothing is invented.
"""
from src.common.evidence import Verdict

LABEL_TEXT = {
    "real": "likely REAL (no strong sign of AI generation or editing)",
    "edited": "likely EDITED (a real image with altered regions)",
    "ai_generated": "likely AI-GENERATED",
    "uncertain": "UNCERTAIN (the evidence is weak or conflicting)",
}


def explain(v: Verdict, max_points: int = 4) -> str:
    head = f"Verdict: {LABEL_TEXT[v.label]}, confidence {v.confidence:.0%}."
    support = sorted((e for e in v.evidence if e.supports == v.label), key=lambda e: -e.strength)
    against = sorted((e for e in v.evidence if e.supports not in (v.label, "neutral")), key=lambda e: -e.strength)
    lines = [head]
    if v.label == "uncertain":
        support = sorted((e for e in v.evidence if e.supports != "neutral"), key=lambda e: -e.strength)
        against = []
    if support:
        lines.append("Main evidence:")
        lines += [f"- {e.description}" for e in support[:max_points]]
    if against:
        lines.append("Evidence pointing the other way:")
        lines += [f"- {e.description}" for e in against[:2]]
    neutral = [e for e in v.evidence if e.supports == "neutral" and e.kind == "metadata"]
    lines += [f"Note: {e.description}" for e in neutral]
    return "\n".join(lines)
