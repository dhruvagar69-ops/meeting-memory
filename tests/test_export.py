import csv

from eval.export_items import export, score
from ingest.parse import Segment, segment_id
from store import db


def make_conn():
    c = db.connect(":memory:")
    texts = [("A (PM)", "Should we add voice recognition?"), ("B (ID)", "I would say no, too expensive."),
             ("A (PM)", "Okay, no voice recognition then.")]
    segs = [Segment(segment_id("M", i), "M", i, s, i * 10.0, i * 10.0 + 5, t) for i, (s, t) in enumerate(texts)]
    db.add_meeting(c, "M", "t", "test", None, segs)
    db.add_item(c, "M", "decision", "No voice recognition", "M-s0001")
    db.add_item(c, "M", "question", "Add voice recognition?", "M-s0000")
    return c


def test_export_and_score_round_trip(tmp_path):
    out = tmp_path / "labels.csv"
    assert export(make_conn(), ["M"], out) == 2

    with open(out, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    dec = next(r for r in rows if r["type"] == "decision")
    assert dec["time"] == "00:10" and dec["cited_text"] == "I would say no, too expensive."
    assert ">>B (ID): I would say no" in dec["context"] and "A (PM): Okay, no voice" in dec["context"]

    rows[0]["verdict"], rows[1]["verdict"] = "supported", "partial"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    res = score(out)
    assert res["unlabelled"] == 0
    assert res["ALL"]["precision_strict"] == 0.5 and res["ALL"]["precision_lenient"] == 1.0
