import json

import pytest

from extract.extract import call_json, chunk_lines, extract_meeting, render_lines
from extract.llm import ExtractionError
from extract.verify import resolve_owner, verify_items
from ingest.parse import Segment, segment_id
from store import db

TEXT = [
    ("A (PM)", "Okay so we agreed to go with the rubber buttons."),
    ("D (ME)", "Hmm."),
    ("B (ID)", "I will send the first mock up by Friday."),
    ("C (UI)", "Should we add a speech recogniser or not?"),
]


@pytest.fixture
def conn():
    c = db.connect(":memory:")
    segs = [Segment(segment_id("M", i), "M", i, spk, float(i * 10), float(i * 10 + 5), t)
            for i, (spk, t) in enumerate(TEXT)]
    db.add_meeting(c, "M", "t", "test", None, segs)
    return c


def reply(**kw):
    return json.dumps(kw)


GOOD = reply(
    decisions=[{"text": "Use rubber buttons", "segment": 0, "quote": "agreed to go with the rubber buttons"}],
    actions=[{"text": "Send first mock up", "owner": "industrial designer", "segment": 2,
              "deadline": "Friday", "quote": "send the first mock up by Friday"}],
    questions=[{"text": "Add speech recogniser?", "segment": 3, "quote": "add a speech recogniser or not"}],
)


def test_good_reply_is_kept_and_stored(conn):
    rep = extract_meeting(conn, "M", lambda s, u: GOOD)
    assert len(rep.items) == 3 and not rep.dropped
    action = next(i for i in rep.items if i.type == "action")
    assert action.owner == "B (ID)" and action.deadline == "Friday"
    stored = db.list_items(conn, type_="action")
    assert stored[0]["cited_segment_id"] == "M-s0002"


def test_backchannel_is_not_sent_to_model(conn):
    rows = db.get_segments(conn, "M")
    assert [i for i, _ in render_lines(rows)] == [0, 2, 3]


def test_hallucinations_are_dropped(conn):
    bad = reply(
        decisions=[
            {"text": "Ghost", "segment": 99, "quote": "whatever"},
            {"text": "Paraphrased", "segment": 0, "quote": "we chose rubber"},
        ],
        actions=[{"text": "Do thing", "owner": "Nobody", "segment": 2,
                  "deadline": "next Tuesday", "quote": "send the first mock up"}],
    )
    rep = extract_meeting(conn, "M", lambda s, u: bad)
    reasons = sorted(d.reason for d in rep.dropped)
    assert reasons == ["cited segment does not exist", "quote not found in cited segment"]
    (action,) = rep.items
    assert action.owner == "Unassigned" and action.deadline is None


def test_lenient_mode_keeps_paraphrased_quote(conn):
    bad = reply(decisions=[{"text": "Paraphrased", "segment": 0, "quote": "we chose rubber"}])
    assert len(extract_meeting(conn, "M", lambda s, u: bad, strict_quotes=False).items) == 1


def test_invalid_json_gets_one_retry_then_fails(conn):
    calls = []
    def flaky(s, u):
        calls.append(u)
        return "not json" if len(calls) == 1 else GOOD
    assert len(extract_meeting(conn, "M", flaky).items) == 3 and len(calls) == 2
    with pytest.raises(ExtractionError):
        call_json(lambda s, u: "still not json", "sys", "user")


def test_fenced_json_and_reruns_replace_items(conn):
    extract_meeting(conn, "M", lambda s, u: "```json\n" + GOOD + "\n```")
    extract_meeting(conn, "M", lambda s, u: GOOD)
    assert len(db.list_items(conn, status=None)) == 3


def test_owner_resolution():
    speakers = ["A (PM)", "B (ID)"]
    assert resolve_owner("A", speakers) == "A (PM)"
    assert resolve_owner("PM", speakers) == "A (PM)"
    assert resolve_owner("Project Manager", speakers) == "A (PM)"
    assert resolve_owner("Rose", speakers) == "Unassigned"
    assert resolve_owner(None, speakers) == "Unassigned"


def test_chunking_overlaps():
    lines = [(i, "x" * 100) for i in range(30)]
    chunks = chunk_lines(lines, max_chars=1000, overlap=3)
    assert len(chunks) > 1 and chunks[1][0][0] == chunks[0][-3][0]
    assert {i for c in chunks for i, _ in c} == set(range(30))


def test_non_object_reply_is_reported():
    kept, dropped = verify_items([1, 2], {}, [])
    assert not kept and dropped[0].reason == "reply was not a JSON object"


def test_all_chunks_failing_keeps_existing_items(conn):
    extract_meeting(conn, "M", lambda s, u: GOOD)
    def down(s, u):
        raise ExtractionError("HTTP 401 (check the API key)")
    with pytest.raises(ExtractionError, match="401"):
        extract_meeting(conn, "M", down)
    assert len(db.list_items(conn, status=None)) == 3


def test_json_with_surrounding_text_is_accepted():
    out = call_json(lambda s, u: "Here you go:\n" + GOOD + "\nHope that helps!", "sys", "user")
    assert len(out["decisions"]) == 1


def test_progress_messages_are_reported(conn):
    msgs = []
    extract_meeting(conn, "M", lambda s, u: GOOD, progress=msgs.append)
    assert any("chunk 1/1" in m and "waiting" in m for m in msgs)
    assert any("done" in m and "kept 3" in m for m in msgs)
