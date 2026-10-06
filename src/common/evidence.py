"""Shared data structures. Every expert returns Evidence items; the fusion step turns them into a Verdict.

The explainer may only cite Evidence items, which keeps explanations grounded in what was measured.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

LABELS = ("real", "edited", "ai_generated", "uncertain")


@dataclass
class Evidence:
    id: str                 # stable key, e.g. "image.ai_score"
    source: str             # which expert produced it
    kind: str               # "score" | "map" | "metadata" | "segments" | "tokens"
    value: Any              # number, string, list, or file path for maps
    supports: str           # which label this pushes towards: real / edited / ai_generated / neutral
    strength: float         # 0..1, how strongly
    description: str        # one human-readable sentence, used by the explainer


@dataclass
class Verdict:
    label: str
    confidence: float
    probabilities: dict[str, float]
    evidence: list[Evidence] = field(default_factory=list)
    explanation: str = ""
    artifacts: dict[str, str] = field(default_factory=dict)   # name -> file path (heatmaps, masks)

    def to_dict(self) -> dict:
        return asdict(self)
