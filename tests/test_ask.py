import json

from ask.ask import NOT_FOUND, answer_question, check_citations, format_answer
from ask.retrieve import fts_query, retrieve
from ingest.parse import Segment, segment_id
from store import db


def make_conn():
    c = db.connect(":memory:")
    for mid, lines in {
        "M1": [("A (PM)", "Welcome everyone."), ("B (ID)", "What about the price?"),
               ("A (PM)", "Target price is twelve point five euros."), ("B (ID)", "Okay, agreed.")],
        "M2": [("C (UI)", "Large buttons for the essentials."), ("D (ME)", "The battery should be kinetic.")],
    }.items():
        segs = [Segment(segment_id(mid, i), mid, i, s, i * 10.0, i * 10.0 + 5, t)
                for i, (s, t) in enumerate(lines)]
        db.add_meeting(c, mid, mid, "test", None, segs)
    return c


def fake(payload):
    calls = []

    def llm(system, user):
        calls.append(user)
        return json.dumps(payload)
    llm.calls = calls
    return llm


def test_fts_query_stems_and_drops_stopwords():
    assert fts_query("What did we decide about the buttons?") == "decid* OR butto*"
    assert fts_query("what is the of") is None


def test_retrieve_merges_neighbours_across_meetings():
    ws = retrieve(make_conn(), "price and buttons")
    assert {w.meeting_id for w in ws} == {"M1", "M2"}
    m1 = next(w for w in ws if w.meeting_id == "M1")
    assert [r["idx"] for r in m1.rows] == [0, 1, 2, 3]  # hits at 1 and 2 merged with neighbours


def test_nothing_found_never_calls_the_model():
    llm = fake({"found": True, "answer": "made up [S1]"})
    a = answer_question(make_conn(), "capital of France", llm)
    assert not a.found and a.text == NOT_FOUND and llm.calls == []


def test_valid_citation_is_verified_and_fabricated_one_removed():
    llm = fake({"found": True, "answer": "A (PM) set the price at 12.5 euros [S1][S9]."})
    a = answer_question(make_conn(), "What is the target price?", llm)
    assert a.found and a.verified and a.removed_citations == [9]
    assert "S9" not in a.text and "[S1]" in a.text
    assert "Target price is twelve point five euros." in format_answer(a)


def test_answer_without_valid_citation_is_unverified():
    a = answer_question(make_conn(), "What is the target price?", fake({"found": True, "answer": "Twelve euros."}))
    assert a.found and not a.verified and "UNVERIFIED" in format_answer(a)


def test_model_can_say_not_found():
    a = answer_question(make_conn(), "What is the target price?", fake({"found": False, "answer": ""}))
    assert not a.found and a.text == NOT_FOUND


def test_check_citations_handles_lists():
    assert check_citations("x [S1, S4] y [12]", {1, 2}) == ("x [S1] y [12]", [1], [4])


def test_per_meeting_keeps_a_quiet_meeting_from_being_crowded_out():
    from ask.retrieve import retrieve
    c = db.connect(":memory:")
    loud = [Segment(segment_id("L", i), "L", i, "A", i * 10.0, i * 10.0 + 5, "button button button design")
            for i in range(0, 20, 2)]
    quiet = [Segment(segment_id("Q", 0), "Q", 0, "B", 0.0, 5.0, "one button was mentioned here")]
    db.add_meeting(c, "L", "L", "t", None, loud)
    db.add_meeting(c, "Q", "Q", "t", None, quiet)
    assert {w.meeting_id for w in retrieve(c, "button", k=3, context=0)} == {"L"}
    assert {w.meeting_id for w in retrieve(c, "button", k=3, context=0, per_meeting=1)} == {"L", "Q"}


def test_broad_questions_get_a_helpful_message_and_do_not_call_the_model():
    from ask.ask import BROAD_QUESTION
    from ask.retrieve import is_broad_question
    for q in ["What was the meeting about?", "Summarize the meeting", "What was discussed in the meeting?"]:
        assert is_broad_question(q)
        llm = fake({"found": True, "answer": "made up [S1]"})
        a = answer_question(make_conn(), q, llm)
        assert not a.found and a.text == BROAD_QUESTION and llm.calls == []
        assert BROAD_QUESTION in format_answer(a)
    assert not is_broad_question("What was decided about the price?")
    assert not is_broad_question("What is the capital of France?")
