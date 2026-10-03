"""Run your hand-written questions through the ask workflow and measure it.

    python -m eval.run_eval --questions eval/questions.json --tag v1 \
        --backend groq --model openai/gpt-oss-120b --reasoning-effort medium
    python -m eval.run_eval ... --expand --tag expand       # compare with/without query expansion
    python -m eval.run_eval --score eval/results_v1.csv     # after filling the `correct` column

questions.json: a list of
  {"id", "question", "gold_answer", "gold_segments": ["ES2008d-s0335", ...],
   "answerable": true/false, "multi_meeting": true/false}
Write the questions and gold segments BEFORE looking at the system's answers.

Automatic metrics (answerable questions):
  retrieval_hit  a gold segment was in the retrieved windows (before the model sees anything)
  citation_hit   a gold segment was inside a window the answer actually cited
Unanswerable questions should come back as "Not found".
You fill the `correct` column yourself: yes / partial / no.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from ask.ask import answer_question, expand_terms
from ask.retrieve import retrieve
from extract.llm import ExtractionError, LLM, make_llm, tokens_used
from store import db

COLUMNS = ["id", "question", "answerable", "multi_meeting", "retrieval_hit", "citation_hit", "found",
           "verified", "answer", "cited", "gold_answer", "gold_segments", "correct", "notes"]


BOOL_COLS = {"answerable", "multi_meeting", "retrieval_hit", "citation_hit", "found", "verified"}


def load_questions(path: Path) -> list[dict]:
    qs = json.loads(Path(path).read_text(encoding="utf-8"))
    for q in qs:
        q.setdefault("gold_segments", [])
        q.setdefault("answerable", True)
        q.setdefault("multi_meeting", False)
    return qs


def _hit(windows, gold) -> bool:
    return any(r["id"] in gold for w in windows for r in w.rows)


def run(conn, questions: list[dict], llm: LLM, *, expand: bool = False, max_windows: int = 6,
        progress=None, skip_ids=(), on_row=None, per_meeting: int = 0) -> list[dict]:
    rows = []
    for n, q in enumerate(questions, 1):
        q = {"gold_segments": [], "answerable": True, "multi_meeting": False, **q}
        if q["id"] in skip_ids:
            continue
        if progress:
            progress(f"  [{n}/{len(questions)}] {q['question']}")
        gold = set(q["gold_segments"])
        terms = expand_terms(llm, q["question"]) if expand else []
        windows = retrieve(conn, q["question"], max_windows=max_windows, extra_terms=terms,
                           per_meeting=per_meeting)
        ans = answer_question(conn, q["question"], llm, max_windows=max_windows, extra_terms=terms,
                              per_meeting=per_meeting)
        row = {
            "id": q["id"], "question": q["question"], "answerable": q["answerable"],
            "multi_meeting": q["multi_meeting"],
            "retrieval_hit": _hit(windows, gold) if q["answerable"] else "",
            "citation_hit": _hit([w for _, w in ans.sources], gold) if q["answerable"] else "",
            "found": ans.found, "verified": ans.verified, "answer": ans.text,
            "cited": "; ".join(f"{w.meeting_id} {int(w.start // 60):02d}:{int(w.start % 60):02d}"
                               for _, w in ans.sources),
            "gold_answer": q.get("gold_answer", ""), "gold_segments": " ".join(q["gold_segments"]),
            "correct": "", "notes": ""}
        rows.append(row)
        if on_row:
            on_row(row)
    return rows


def _rate(rows, key) -> str:
    return f"{sum(1 for r in rows if r[key] is True)}/{len(rows)}" if rows else "n/a"


def summarize(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    multi = [r for r in ans if r["multi_meeting"]]
    return {
        "answerable": len(ans), "retrieval_hit": _rate(ans, "retrieval_hit"),
        "citation_hit": _rate(ans, "citation_hit"),
        "multi_meeting_citation_hit": _rate(multi, "citation_hit"),
        "answerable_but_not_found": sum(1 for r in ans if not r["found"]),
        "unverified_answers": sum(1 for r in rows if r["found"] and not r["verified"]),
        "unanswerable": len(unans),
        "unanswerable_correctly_not_found": f"{sum(1 for r in unans if not r['found'])}/{len(unans)}" if unans else "n/a",
    }


def read_csv_rows(path: Path) -> list[dict]:
    """Load a results CSV written by write_csv (keeps any `correct` labels you already filled in)."""
    conv = {"True": True, "False": False}
    with open(path, newline="", encoding="utf-8-sig") as f:
        return [{k: (conv.get(v, v) if k in BOOL_COLS else v) for k, v in r.items()} for r in csv.DictReader(f)]


def write_csv(rows: list[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


def score_correct(path: Path) -> dict:
    counts = {"yes": 0, "partial": 0, "no": 0, "unlabelled": 0}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            v = (r.get("correct") or "").strip().lower()
            counts[v if v in counts else "unlabelled"] += 1
    n = counts["yes"] + counts["partial"] + counts["no"]
    counts["accuracy_strict"] = round(counts["yes"] / n, 2) if n else None
    counts["accuracy_lenient"] = round((counts["yes"] + counts["partial"]) / n, 2) if n else None
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", default="eval/questions.json")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--out")
    ap.add_argument("--score", metavar="CSV", help="score the `correct` column of a results CSV")
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
                    help="also take this many best matches from every meeting")
    ap.add_argument("--expand", action="store_true")
    ap.add_argument("--resume", action="store_true", help="continue a stopped run, skipping finished questions")
    args = ap.parse_args()

    if args.score:
        print(score_correct(Path(args.score)))
        return
    if not args.model:
        ap.error("pass --model <name> or set MM_MODEL")
    out = Path(args.out or f"eval/results_{args.tag}.csv")
    done = read_csv_rows(out) if args.resume and out.exists() else []
    all_rows = list(done)

    def save(row: dict) -> None:
        all_rows.append(row)
        write_csv(all_rows, out)  # saved after every question, so a stop loses nothing

    if done:
        print(f"Resuming: {len(done)} questions already finished in {out}")
    stopped = None
    try:
        llm = make_llm(args.backend, args.model, args.host, args.base_url, log=lambda m: print(m, flush=True),
                       max_tokens=args.max_tokens, reasoning_effort=args.reasoning_effort)
        run(db.connect(args.db), load_questions(args.questions), llm, expand=args.expand,
            max_windows=args.max_windows, per_meeting=args.per_meeting,
            progress=lambda m: print(m, flush=True),
            skip_ids={r["id"] for r in done}, on_row=save)
    except ExtractionError as exc:
        stopped = exc
    print(f"\nwrote {len(all_rows)} rows to {out}")
    if args.backend != "ollama":
        print(f"tokens used by this run: {tokens_used()}")
    if all_rows:
        for k, v in summarize(all_rows).items():
            print(f"  {k}: {v}")
    if stopped:
        raise SystemExit(f"\nStopped early: {stopped}\nYour {len(all_rows)} finished answers are saved. "
                         "Run the same command again with --resume to continue.")
    print("Now open the CSV and fill the `correct` column (yes / partial / no), then run --score.")


if __name__ == "__main__":
    main()
