"""Extract decisions / action items / open questions from stored meetings.

    python -m extract.extract ES2008a ES2008b --model <ollama-model>
    python -m extract.extract ES2008b --backend groq --model <groq-model>   # needs GROQ_API_KEY
    python -m extract.extract ES2008a --lenient --show-dropped
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field

from extract.llm import LLM, ExtractionError, make_llm, ollama
from extract.prompts import SYSTEM, build_user
from extract.verify import BACKCHANNEL, Drop, Item, norm, verify_items
from store import db


@dataclass
class Report:
    meeting_id: str
    items: list[Item] = field(default_factory=list)
    dropped: list[Drop] = field(default_factory=list)
    chunks: int = 0


def render_lines(rows) -> list[tuple[int, str]]:
    """(segment idx, prompt line) for every segment that is not just a backchannel."""
    return [(r["idx"], f"[{r['idx']}] {r['speaker']}: {r['text']}")
            for r in rows if norm(r["text"]) not in BACKCHANNEL]


def chunk_lines(lines: list[tuple[int, str]], max_chars: int = 12000,
                overlap: int = 6) -> list[list[tuple[int, str]]]:
    chunks, cur, size = [], [], 0
    for ln in lines:
        if cur and size + len(ln[1]) > max_chars:
            chunks.append(cur)
            cur = cur[-overlap:]
            size = sum(len(x[1]) for x in cur)
        cur.append(ln)
        size += len(ln[1])
    if cur:
        chunks.append(cur)
    return chunks


def call_json(llm: LLM, system: str, user: str) -> dict:
    """Ask for JSON; on invalid output retry once with a corrective message."""
    last = ""
    for attempt in range(2):
        prompt = user if attempt == 0 else (
            user + "\n\nYour previous reply was not valid JSON. "
                   "Reply with ONLY the JSON object, nothing else.")
        last = llm(system, prompt)
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", last.strip())
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            continue
    raise ExtractionError(f"model returned invalid JSON twice: {last[:120]!r}")


def extract_meeting(conn, meeting_id: str, llm: LLM, *, strict_quotes: bool = True,
                    max_chars: int = 12000) -> Report:
    rows = db.get_segments(conn, meeting_id)
    if not rows:
        raise ValueError(f"No segments stored for {meeting_id}. Load it with ingest.add_meeting first.")
    speakers = sorted({r["speaker"] for r in rows})
    by_idx = {r["idx"]: r for r in rows}
    report = Report(meeting_id)

    seen: set[tuple[str, str]] = set()
    for chunk in chunk_lines(render_lines(rows), max_chars):
        report.chunks += 1
        try:
            raw = call_json(llm, SYSTEM, build_user(meeting_id, speakers, [l for _, l in chunk]))
        except ExtractionError as exc:
            report.dropped.append(Drop("chunk", "", str(exc)))
            continue
        kept, dropped = verify_items(raw, by_idx, speakers, strict_quotes)
        report.dropped += dropped
        for it in kept:
            key = (it.type, norm(it.text))
            if key not in seen:
                seen.add(key)
                report.items.append(it)

    errors = [d for d in report.dropped if d.type == "chunk"]
    if report.chunks and len(errors) == report.chunks:  # every call failed: keep old items
        raise ExtractionError(errors[0].reason)
    db.clear_items(conn, meeting_id)
    for it in report.items:
        db.add_item(conn, meeting_id, it.type, it.text, it.segment_id, it.owner, it.deadline)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("meetings", nargs="+")
    ap.add_argument("--db", default=db.DEFAULT_DB)
    ap.add_argument("--backend", default=os.environ.get("MM_BACKEND", "ollama"),
                    choices=["ollama", "groq", "openrouter", "openai"],
                    help="ollama (local), groq / openrouter (hosted, needs GROQ_API_KEY / "
                         "OPENROUTER_API_KEY), openai (any OpenAI-compatible server, needs --base-url)")
    ap.add_argument("--model", default=os.environ.get("MM_MODEL"), help="model name (or set MM_MODEL)")
    ap.add_argument("--host", default="http://localhost:11434", help="Ollama host")
    ap.add_argument("--base-url", help="for --backend openai")
    ap.add_argument("--lenient", action="store_true", help="do not require word-for-word quotes")
    ap.add_argument("--show-dropped", action="store_true")
    args = ap.parse_args()
    if not args.model:
        ap.error("pass --model <name> or set MM_MODEL")

    conn = db.connect(args.db)
    try:
        llm = make_llm(args.backend, args.model, args.host, args.base_url)
    except ExtractionError as exc:
        ap.error(str(exc))
    for mid in args.meetings:
        try:
            rep = extract_meeting(conn, mid, llm, strict_quotes=not args.lenient)
        except ExtractionError as exc:
            sys.exit(f"{mid}: extraction failed, nothing was changed. {exc}")
        proposed = len(rep.items) + len(rep.dropped)
        print(f"\n=== {mid}: kept {len(rep.items)} of {proposed} proposed "
              f"({rep.chunks} chunk(s)) ===")
        for typ in ("decision", "action", "question"):
            for it in (i for i in rep.items if i.type == typ):
                seg = db.get_segment(conn, it.segment_id)
                who = f" [{it.owner}]" if it.owner else ""
                dl = f" (by {it.deadline})" if it.deadline else ""
                print(f"{typ.upper():9}{who} {it.text}{dl}\n          "
                      f"cite {it.segment_id} @{int(seg['start_s'] // 60):02d}:{int(seg['start_s'] % 60):02d}"
                      f"  \"{it.quote}\"")
        if rep.dropped:
            print("dropped:", dict(Counter(d.reason for d in rep.dropped)))
            if args.show_dropped:
                for d in rep.dropped:
                    print(f"  - {d.type}: {d.text[:70]} ({d.reason})")


if __name__ == "__main__":
    main()
