"""Extract decisions / action items / open questions from stored meetings.

    python -m extract.extract ES2008a ES2008b --model <ollama-model>
    python -m extract.extract ES2008b --backend groq --model <groq-model>   # needs GROQ_API_KEY
    python -m extract.extract --backend groq --list-models      # which model names can I use?
    python -m extract.extract ES2008a --lenient --show-dropped
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field

from extract.llm import LLM, ExtractionError, list_models, make_llm, ollama, tokens_used
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
        for candidate in (text, text[text.find("{"):text.rfind("}") + 1]):  # tolerate text around the JSON
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
    raise ExtractionError(f"model returned invalid JSON twice: {last[:120]!r}")


def extract_meeting(conn, meeting_id: str, llm: LLM, *, strict_quotes: bool = True,
                    max_chars: int = 12000, progress=None) -> Report:
    rows = db.get_segments(conn, meeting_id)
    if not rows:
        raise ValueError(f"No segments stored for {meeting_id}. Load it with ingest.add_meeting first.")
    speakers = sorted({r["speaker"] for r in rows})
    by_idx = {r["idx"]: r for r in rows}
    report = Report(meeting_id)

    seen: set[tuple[str, str]] = set()
    say = progress or (lambda msg: None)
    all_chunks = chunk_lines(render_lines(rows), max_chars)
    for n, chunk in enumerate(all_chunks, 1):
        report.chunks += 1
        say(f"  chunk {n}/{len(all_chunks)}: {len(chunk)} lines, waiting for the model ...")
        started = time.time()
        try:
            raw = call_json(llm, SYSTEM, build_user(meeting_id, speakers, [l for _, l in chunk]))
        except ExtractionError as exc:
            report.dropped.append(Drop("chunk", "", str(exc)))
            say(f"  chunk {n}/{len(all_chunks)}: FAILED after {time.time() - started:.0f}s")
            continue
        kept, dropped = verify_items(raw, by_idx, speakers, strict_quotes)
        say(f"  chunk {n}/{len(all_chunks)}: done in {time.time() - started:.0f}s, "
            f"kept {len(kept)}, dropped {len(dropped)}")
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
    ap.add_argument("meetings", nargs="*")
    ap.add_argument("--db", default=db.DEFAULT_DB)
    ap.add_argument("--backend", default=os.environ.get("MM_BACKEND", "ollama"),
                    choices=["ollama", "groq", "openrouter", "openai"],
                    help="ollama (local), groq / openrouter (hosted, needs GROQ_API_KEY / "
                         "OPENROUTER_API_KEY), openai (any OpenAI-compatible server, needs --base-url)")
    ap.add_argument("--model", default=os.environ.get("MM_MODEL"), help="model name (or set MM_MODEL)")
    ap.add_argument("--host", default="http://localhost:11434", help="Ollama host")
    ap.add_argument("--base-url", help="for --backend openai")
    ap.add_argument("--max-tokens", type=int, default=6000,
                    help="max output tokens per call (reasoning models need room to think)")
    ap.add_argument("--reasoning-effort", choices=["low", "medium", "high"],
                    help="for reasoning models such as openai/gpt-oss-*; lower is faster")
    ap.add_argument("--chunk-chars", type=int, default=12000,
                    help="max characters of transcript per model call; smaller chunks can find more items")
    ap.add_argument("--list-models", action="store_true",
                    help="print the model names your API key can use, then exit")
    ap.add_argument("--lenient", action="store_true", help="do not require word-for-word quotes")
    ap.add_argument("--show-dropped", action="store_true")
    args = ap.parse_args()
    if args.list_models:
        if args.backend == "ollama":
            ap.error("for Ollama run `ollama list` instead")
        try:
            print("\n".join(list_models(args.backend, args.base_url)))
        except ExtractionError as exc:
            ap.error(str(exc))
        return
    if not args.meetings:
        ap.error("give at least one meeting id, e.g. ES2008b")
    if not args.model:
        ap.error("pass --model <name> or set MM_MODEL")

    conn = db.connect(args.db)
    try:
        llm = make_llm(args.backend, args.model, args.host, args.base_url,
                       log=lambda m: print(m, flush=True), max_tokens=args.max_tokens,
                       reasoning_effort=args.reasoning_effort)
    except ExtractionError as exc:
        ap.error(str(exc))
    for mid in args.meetings:
        try:
            print(f"{mid}: extracting (this can take a few minutes; Ctrl+C to stop) ...", flush=True)
            rep = extract_meeting(conn, mid, llm, strict_quotes=not args.lenient,
                                  max_chars=args.chunk_chars,
                                  progress=lambda m: print(m, flush=True))
        except ExtractionError as exc:
            sys.exit(f"{mid}: extraction failed, nothing was changed. {exc}")
        chunk_errors = [d for d in rep.dropped if d.type == "chunk"]
        other_drops = [d for d in rep.dropped if d.type != "chunk"]
        proposed = len(rep.items) + len(other_drops)
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
        if chunk_errors:
            print(f"WARNING: {len(chunk_errors)} of {rep.chunks} chunk(s) failed, so these results are incomplete. "
                  f"First error: {chunk_errors[0].reason.strip()[:300]}")
        if other_drops:
            print("dropped:", dict(Counter(d.reason for d in other_drops)))
            if args.show_dropped:
                for d in other_drops:
                    print(f"  - {d.type}: {d.text[:70]} ({d.reason})")
    if args.backend != "ollama":
        print(f"\ntokens used by this run: {tokens_used()}")


if __name__ == "__main__":
    main()
