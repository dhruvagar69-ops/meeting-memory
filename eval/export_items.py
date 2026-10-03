"""Export extracted items to a CSV for hand-labelling, and score the labels.

    python -m eval.export_items ES2008b                 # writes eval/labels_ES2008b.csv
    python -m eval.export_items ES2008b ES2008c --out eval/labels_all.csv
    python -m eval.export_items --score eval/labels_ES2008b.csv

Fill the `verdict` column with one of: supported, partial, unsupported
  supported   - the cited moment clearly shows this decision/action/question
  partial     - related, but the item says more than the moment shows (or the owner is wrong)
  unsupported - the cited moment does not show it, or it is a question/request/already-done
                thing presented as a decision/open action
Open the CSV in Excel; the file is saved so Excel reads it correctly.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from store import db

COLUMNS = ["item_id", "meeting_id", "type", "owner", "deadline", "text", "cited_segment_id",
           "time", "speaker", "cited_text", "context", "verdict", "notes"]
VERDICTS = ("supported", "partial", "unsupported")


def _mmss(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


def export(conn, meetings: list[str], out_path: Path, context: int = 3) -> int:
    rows = []
    for mid in meetings:
        for it in db.list_items(conn, meeting_id=mid, status=None):
            near = db.neighbours(conn, it["cited_segment_id"], context, context)
            ctx = " | ".join(f"{'>>' if r['id'] == it['cited_segment_id'] else ''}{r['speaker']}: {r['text']}"
                             for r in near)
            rows.append({
                "item_id": it["id"], "meeting_id": mid, "type": it["type"], "owner": it["owner"] or "",
                "deadline": it["deadline"] or "", "text": it["text"],
                "cited_segment_id": it["cited_segment_id"], "time": _mmss(it["cited_start"]),
                "speaker": it["cited_speaker"], "cited_text": it["cited_text"], "context": ctx,
                "verdict": "", "notes": ""})
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:  # BOM so Excel shows UTF-8 correctly
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def score(csv_path: Path) -> dict:
    """Precision from hand labels. strict = supported only; lenient = supported + partial."""
    by_type: dict[str, Counter] = defaultdict(Counter)
    unlabelled = 0
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            v = (row.get("verdict") or "").strip().lower()
            if v in VERDICTS:
                by_type[row["type"]][v] += 1
                by_type["ALL"][v] += 1
            else:
                unlabelled += 1
    out = {"unlabelled": unlabelled}
    for typ, c in by_type.items():
        n = sum(c.values())
        out[typ] = {"labelled": n, "supported": c["supported"], "partial": c["partial"],
                    "unsupported": c["unsupported"],
                    "precision_strict": round(c["supported"] / n, 2),
                    "precision_lenient": round((c["supported"] + c["partial"]) / n, 2)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("meetings", nargs="*")
    ap.add_argument("--db", default=db.DEFAULT_DB)
    ap.add_argument("--out")
    ap.add_argument("--score", metavar="CSV", help="score a labelled CSV instead of exporting")
    args = ap.parse_args()

    if args.score:
        res = score(Path(args.score))
        print(f"unlabelled rows: {res.pop('unlabelled')}")
        for typ, r in res.items():
            print(f"{typ:9} labelled {r['labelled']:3}  supported {r['supported']}  partial {r['partial']}  "
                  f"unsupported {r['unsupported']}  precision strict {r['precision_strict']:.2f}  "
                  f"lenient {r['precision_lenient']:.2f}")
        return
    if not args.meetings:
        ap.error("give at least one meeting id, or use --score")
    out = Path(args.out or f"eval/labels_{'_'.join(args.meetings)}.csv")
    n = export(db.connect(args.db), args.meetings, out)
    print(f"wrote {n} items to {out}")


if __name__ == "__main__":
    main()
