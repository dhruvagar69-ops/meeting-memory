"""Grounding checks. A model proposal is kept only if it survives these.

- the cited segment must exist
- the quote must appear word-for-word in the cited segment (strict mode)
- an owner must be a real speaker in this meeting, otherwise "Unassigned"
- a deadline is kept only if it is actually spoken in the cited segment
"""
from __future__ import annotations

import re
from dataclasses import dataclass

KEYS = (("decisions", "decision"), ("actions", "action"), ("questions", "question"))
ROLE_NAMES = {  # AMI role codes -> spoken names
    "PM": "project manager", "ID": "industrial designer",
    "UI": "user interface", "ME": "marketing expert",
}
BACKCHANNEL = {"mm", "hmm", "hm", "um", "uh", "mhm", "em", "oh", "uh huh", "mm hmm",
               "yeah", "yes", "okay", "ok", "right", "no", "well", "so"}

_PUNCT = re.compile(r"[^\w\s]")


def norm(s: str) -> str:
    return " ".join(_PUNCT.sub(" ", s.lower().replace("_", " ")).split())


@dataclass(frozen=True)
class Item:
    type: str
    text: str
    owner: str | None
    deadline: str | None
    segment_id: str
    quote: str


@dataclass(frozen=True)
class Drop:
    type: str
    text: str
    reason: str


def _as_int(v) -> int | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    m = re.search(r"\d+", str(v)) if v is not None else None
    return int(m.group()) if m else None


def quote_ok(quote: str, segment_text: str) -> bool:
    q, s = norm(quote), norm(segment_text)
    return bool(q) and q in s and (len(q.split()) >= 2 or q == s)


def resolve_owner(owner, speakers: list[str]) -> str:
    on = norm(str(owner)) if owner else ""
    if not on:
        return "Unassigned"
    for label in speakers:
        m = re.match(r"^(\S+)\s*\((.+)\)$", label)
        letter, role = (m.group(1), m.group(2)) if m else (label, "")
        cands = {norm(label), norm(letter), norm(role), norm(ROLE_NAMES.get(role, ""))} - {""}
        if on in cands:
            return label
    return "Unassigned"


def verify_items(raw: dict, segs_by_idx: dict, speakers: list[str],
                 strict_quotes: bool = True) -> tuple[list[Item], list[Drop]]:
    kept: list[Item] = []
    dropped: list[Drop] = []
    if not isinstance(raw, dict):
        return kept, [Drop("all", str(raw)[:80], "reply was not a JSON object")]
    for key, typ in KEYS:
        entries = raw.get(key) or []
        if not isinstance(entries, list):
            continue
        for e in entries:
            if not isinstance(e, dict):
                dropped.append(Drop(typ, str(e)[:80], "item was not an object"))
                continue
            text = str(e.get("text") or "").strip()
            seg = segs_by_idx.get(_as_int(e.get("segment")))
            quote = str(e.get("quote") or "").strip()
            if not text:
                dropped.append(Drop(typ, "", "empty text"))
            elif seg is None:
                dropped.append(Drop(typ, text, "cited segment does not exist"))
            elif strict_quotes and not quote_ok(quote, seg["text"]):
                dropped.append(Drop(typ, text, "quote not found in cited segment"))
            else:
                owner = resolve_owner(e.get("owner"), speakers) if typ != "question" else None
                dl = str(e.get("deadline") or "").strip()
                deadline = dl if typ == "action" and dl and norm(dl) in norm(seg["text"]) else None
                kept.append(Item(typ, text, owner, deadline, seg["id"], quote))
    return kept, dropped
