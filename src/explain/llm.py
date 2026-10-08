"""Explainer v2: an instruction-tuned LLM rewrites the evidence into plain language, then a faithfulness
check verifies every sentence against the evidence. Unfaithful output falls back to the template explainer.

Grounding idea from "XAI-Grounded Explanation Generation for Speech Deepfake Detection" (arXiv 2606.16137):
the LLM never sees the raw media, only measured evidence, so it cannot invent visual or audio "clues".
The faithfulness rules follow the omitted/irrelevant-evidence failure modes in arXiv 2608.20913.
"""
from __future__ import annotations

import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common.device import get_device
from src.common.evidence import Verdict
from src.explain.template import LABEL_TEXT, explain as template_explain

DEFAULT_LLM = os.environ.get("BTP_EXPLAINER_LLM", "Qwen/Qwen2.5-1.5B-Instruct")

PROMPT = """You explain the result of a media-forensics system to a non-expert.
Verdict: {label} (confidence {conf:.0%}).
Evidence (id: fact):
{facts}

Write 3 to 5 short sentences explaining why the system reached this verdict.
Rules:
- Use ONLY the facts above. Do not add any other clue, number or detail.
- End every sentence with the id of the fact it uses in square brackets, e.g. [image.ai_score].
- If evidence points the other way, mention it.
- If the verdict is uncertain, say what is missing or conflicting."""

_NUM = re.compile(r"\d+(?:\.\d+)?")


class LLMExplainer:
    def __init__(self, model: str = DEFAULT_LLM, device: str = "auto", tok=None, llm=None):
        self.device = get_device(device)
        self.tok = tok or AutoTokenizer.from_pretrained(model)
        self.llm = llm or AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32).to(self.device).eval()

    @torch.no_grad()
    def generate(self, v: Verdict) -> str:
        facts = "\n".join(f"{e.id}: {e.description}" for e in v.evidence)
        msgs = [{"role": "user", "content": PROMPT.format(label=LABEL_TEXT[v.label], conf=v.confidence, facts=facts)}]
        # render to text, then tokenize: works across transformers 4.x and 5.x (5.x returns a dict here)
        prompt = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        inputs = self.tok(prompt, return_tensors="pt").to(self.device)
        out = self.llm.generate(**inputs, max_new_tokens=220, do_sample=False)
        return self.tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def explain(self, v: Verdict) -> tuple[str, dict]:
        draft = self.generate(v)
        ok, report = check_faithfulness(draft, v)
        if not ok:
            return template_explain(v), {"used": "template_fallback", **report, "rejected_draft": draft}
        clean = re.sub(r"\s*\[[\w.]+\]", "", draft)
        return f"Verdict: {LABEL_TEXT[v.label]}, confidence {v.confidence:.0%}.\n{clean}", {"used": "llm", **report}


def check_faithfulness(text: str, v: Verdict) -> tuple[bool, dict]:
    """Every sentence must cite >=1 real evidence id, and every number in it must appear in the cited facts."""
    ev = {e.id: f"{e.description} {e.value}" for e in v.evidence}
    sents = [s.strip() for s in re.split(r"(?<=[.!?\]])\s+", text) if len(s.strip()) > 3]
    problems = []
    for s in sents:
        cited = re.findall(r"\[([\w.]+)\]", s)
        if not cited:
            problems.append(f"no citation: {s[:60]}")
            continue
        unknown = [c for c in cited if c not in ev]
        if unknown:
            problems.append(f"unknown id {unknown}: {s[:60]}")
            continue
        allowed = set(_NUM.findall(" ".join(ev[c] for c in cited)))
        body = re.sub(r"\[[\w.]+\]", "", s)
        bad_nums = [n for n in _NUM.findall(body) if n not in allowed]
        if bad_nums:
            problems.append(f"unsupported numbers {bad_nums}: {s[:60]}")
    supported = {e.id for e in v.evidence if e.supports == v.label}
    cited_all = set(re.findall(r"\[([\w.]+)\]", text))
    if supported and not (supported & cited_all):
        problems.append("omits all evidence supporting the verdict")
    ok = bool(sents) and not problems
    return ok, {"sentences": len(sents), "problems": problems, "faithful": ok}
