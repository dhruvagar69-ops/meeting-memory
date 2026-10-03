"""Ask a question across all stored meetings and get a cited answer.

    python -m ask.ask "What was decided about the price?" --backend groq --model openai/gpt-oss-120b

Flow: retrieve windows -> (nothing found? say so, never call the model) ->
model answers from the numbered sources only -> citations to sources that were
not provided are removed -> an answer with no valid citation is flagged unverified.
"""
from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass, field

from ask.retrieve import Window, retrieve
from extract.extract import call_json
from extract.llm import ExtractionError, LLM, make_llm
from store import db

NOT_FOUND = "Not found in these meetings."

SYSTEM = """You answer questions about meetings using ONLY the numbered sources provided.

Each source is a short stretch of a meeting transcript, labelled like [S1].
Rules:
- Use only what the sources say. Never use outside knowledge and never guess.
- After every claim, cite the source(s) it came from, like [S1] or [S2][S3].
- Say who said it (speaker labels like "A (PM)") when it helps.
- If the sources do not contain the answer, set "found" to false.
- Keep the answer to 1-4 sentences.

Reply with ONLY a JSON object: {"found": true, "answer": "... [S1]"}  or  {"found": false, "answer": ""}"""


EXPAND_SYSTEM = """You help search a meeting transcript. Given a question, list up to 8 single words that people
might actually have said when discussing it, including synonyms and related words
(for "price" you might list cost, euro, expensive, budget, pay).
Reply with ONLY a JSON object: {"terms": ["...", "..."]}"""


def expand_terms(llm: LLM, question: str) -> list[str]:
    """Extra search words from the model. Any failure just means no expansion."""
    try:
        raw = call_json(llm, EXPAND_SYSTEM, question)
    except ExtractionError:
        return []
    terms = raw.get("terms", []) if isinstance(raw, dict) else []
    return [t for t in terms if isinstance(t, str)][:8]


@dataclass
class Answer:
    question: str
    found: bool
    text: str
    verified: bool = False
    sources: list[tuple[int, Window]] = field(default_factory=list)  # (S number, window) actually cited
    removed_citations: list[int] = field(default_factory=list)


def _mmss(sec: float) -> str:
    return f"{int(sec // 60):02d}:{int(sec % 60):02d}"


def render_sources(windows: list[Window]) -> str:
    blocks = []
    for n, w in enumerate(windows, 1):
        lines = "\n".join(f"  {r['speaker']}: {r['text']}" for r in w.rows)
        blocks.append(f"[S{n}] meeting {w.meeting_id}, {_mmss(w.start)}-{_mmss(w.end)}\n{lines}")
    return "\n\n".join(blocks)


def check_citations(text: str, valid: set[int]) -> tuple[str, list[int], list[int]]:
    """Keep only citations to provided sources. Returns (clean text, cited, removed)."""
    cited: list[int] = []
    removed: list[int] = []

    def fix(m: re.Match) -> str:
        ids = [int(x) for x in re.findall(r"S(\d+)", m.group(1))]
        if not ids:
            return m.group(0)
        good = [i for i in ids if i in valid]
        removed.extend(i for i in ids if i not in valid)
        cited.extend(good)
        return "[" + ", ".join(f"S{i}" for i in good) + "]" if good else ""

    clean = re.sub(r"\[([^\]]*)\]", fix, text)
    return re.sub(r"[ \t]{2,}", " ", clean).strip(), sorted(set(cited)), removed


def answer_question(conn, question: str, llm: LLM, *, k: int = 12, context: int = 1,
                    max_windows: int = 6, expand: bool = False, extra_terms=None,
                    per_meeting: int = 0) -> Answer:
    if extra_terms is None:
        extra_terms = expand_terms(llm, question) if expand else []
    windows = retrieve(conn, question, k=k, context=context, max_windows=max_windows, per_meeting=per_meeting,
                       extra_terms=extra_terms)
    if not windows:  # nothing relevant: never call the model, so it cannot invent an answer
        return Answer(question, False, NOT_FOUND)

    user = f"Question: {question}\n\nSources:\n\n{render_sources(windows)}"
    raw = call_json(llm, SYSTEM, user)
    if not raw.get("found") or not str(raw.get("answer", "")).strip():
        return Answer(question, False, NOT_FOUND)

    text, cited, removed = check_citations(str(raw["answer"]), set(range(1, len(windows) + 1)))
    used = [(n, windows[n - 1]) for n in cited]
    return Answer(question, True, text, verified=bool(cited), sources=used, removed_citations=removed)


def format_answer(a: Answer) -> str:
    out = [a.text if a.found else NOT_FOUND]
    if a.found and not a.verified:
        out.append("\nUNVERIFIED: the model gave no valid citation, so treat this as a guess.")
    if a.removed_citations:
        out.append(f"(removed citations to sources that were not provided: {a.removed_citations})")
    if a.sources:
        out.append("\nSources:")
        for n, w in a.sources:
            out.append(f"[S{n}] {w.meeting_id} {_mmss(w.start)}-{_mmss(w.end)}")
            out.extend(f"    {r['speaker']}: {r['text']}" for r in w.rows)
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question")
    ap.add_argument("--db", default=db.DEFAULT_DB)
    ap.add_argument("--backend", default=os.environ.get("MM_BACKEND", "ollama"),
                    choices=["ollama", "groq", "openrouter", "openai"])
    ap.add_argument("--model", default=os.environ.get("MM_MODEL"))
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--base-url")
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--reasoning-effort", choices=["low", "medium", "high"])
    ap.add_argument("--max-windows", type=int, default=6)
    ap.add_argument("--per-meeting", type=int, default=0,
                    help="also take this many best matches from every meeting (helps cross-meeting questions)")
    ap.add_argument("--expand", action="store_true",
                    help="ask the model for extra search words (synonyms) first; costs one more call")
    args = ap.parse_args()
    if not args.model:
        ap.error("pass --model <name> or set MM_MODEL")
    try:
        llm = make_llm(args.backend, args.model, args.host, args.base_url,
                       log=lambda m: print(m, flush=True), max_tokens=args.max_tokens,
                       reasoning_effort=args.reasoning_effort)
        print(format_answer(answer_question(db.connect(args.db), args.question, llm,
                                            max_windows=args.max_windows, expand=args.expand,
                                            per_meeting=args.per_meeting)))
    except ExtractionError as exc:
        raise SystemExit(f"Could not answer: {exc}")


if __name__ == "__main__":
    main()
