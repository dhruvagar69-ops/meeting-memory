"""Find the stretches of conversation most relevant to a question (BM25 over
segment text, with crude prefix stemming so "buttons" also finds "button").

Hits are expanded with neighbouring segments and merged into windows, so an
answer can see who said what just before and after the matching line.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_STOP = {
    "the", "a", "an", "of", "to", "and", "or", "in", "on", "for", "is", "it", "we", "did", "do",
    "what", "about", "that", "this", "was", "were", "who", "which", "when", "where", "how", "why",
    "there", "they", "them", "with", "from", "have", "has", "had", "are", "you", "your", "our",
    "say", "said", "tell", "does", "would", "should", "could", "any", "all", "be", "been", "me",
    "my", "i", "at", "by", "as", "if", "so", "not", "can", "will", "than", "then", "also", "into",
    "meeting", "meetings", "discuss", "discussed", "mention", "mentioned",
}


_BROAD = {"summarize", "summarise", "summary", "overview", "recap", "highlights", "gist", "main",
          "points", "topics", "topic", "everything", "anything", "happened", "going"}


def is_broad_question(question: str) -> bool:
    """True when the question has no specific topic to search for, e.g. "What was the meeting about?"
    or "Summarize the meeting". Keyword search cannot answer these."""
    words = [t for t in re.findall(r"[a-z0-9]+", question.lower()) if t not in _STOP and len(t) >= 3]
    return all(w in _BROAD for w in words)


@dataclass
class Window:
    meeting_id: str
    rows: list            # sqlite rows in order
    score: float          # best (lowest) bm25 in the window

    @property
    def start(self) -> float:
        return self.rows[0]["start_s"]

    @property
    def end(self) -> float:
        return self.rows[-1]["end_s"]


def fts_query(question: str, extra: tuple | list = ()) -> str | None:
    """OR of prefix terms, e.g. 'which buttons?' -> 'butto*'. None if nothing searchable.
    `extra` adds more search words (e.g. synonyms)."""
    terms = []
    for tok in re.findall(r"[a-z0-9]+", (question + " " + " ".join(extra)).lower()):
        if tok in _STOP or len(tok) < 3:
            continue
        stem = tok if len(tok) <= 4 else tok[:5]
        if stem not in terms:
            terms.append(stem)
    return " OR ".join(f"{t}*" for t in terms) if terms else None


def retrieve(conn, question: str, k: int = 12, context: int = 1, max_windows: int = 6,
             extra_terms: tuple | list = (), per_meeting: int = 0) -> list[Window]:
    """per_meeting > 0 also takes that many best hits from EVERY meeting, so a question about
    several meetings is not crowded out by the one meeting with the most matches."""
    match = fts_query(question, extra_terms)
    if not match:
        return []
    hits = conn.execute(
        """SELECT s.meeting_id, s.idx, bm25(segments_fts) AS score
           FROM segments_fts JOIN segments s ON s.rowid = segments_fts.rowid
           WHERE segments_fts MATCH ? ORDER BY score LIMIT ?""", (match, k)).fetchall()
    hits = list(hits)
    if per_meeting > 0:
        for (mid,) in conn.execute("SELECT DISTINCT meeting_id FROM segments").fetchall():
            hits += conn.execute(
                """SELECT s.meeting_id, s.idx, bm25(segments_fts) AS score
                   FROM segments_fts JOIN segments s ON s.rowid = segments_fts.rowid
                   WHERE segments_fts MATCH ? AND s.meeting_id = ? ORDER BY score LIMIT ?""",
                (match, mid, per_meeting)).fetchall()
    if not hits:
        return []

    # merge hit ranges per meeting
    spans: dict[str, list[list]] = {}
    for h in sorted(hits, key=lambda r: (r["meeting_id"], r["idx"])):
        lo, hi = max(0, h["idx"] - context), h["idx"] + context
        cur = spans.setdefault(h["meeting_id"], [])
        if cur and lo <= cur[-1][1] + 1:
            cur[-1][1] = max(cur[-1][1], hi)
            cur[-1][2] = min(cur[-1][2], h["score"])
        else:
            cur.append([lo, hi, h["score"]])

    windows = []
    for mid, ranges in spans.items():
        for lo, hi, score in ranges:
            rows = conn.execute(
                "SELECT * FROM segments WHERE meeting_id = ? AND idx BETWEEN ? AND ? ORDER BY idx",
                (mid, lo, hi)).fetchall()
            if rows:
                windows.append(Window(mid, rows, score))
    windows.sort(key=lambda w: w.score)
    return windows[:max_windows]
