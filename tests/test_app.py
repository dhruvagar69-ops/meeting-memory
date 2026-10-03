import json
from pathlib import Path
from unittest import mock

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from ingest.parse import Segment, segment_id  # noqa: E402
from store import db  # noqa: E402

# app.py lives in the project root; find it whether this file is in tests/ or in the root itself
_HERE = Path(__file__).resolve().parent
APP = next(d / "app.py" for d in (_HERE, _HERE.parent) if (d / "app.py").is_file())


@pytest.fixture
def db_file(tmp_path, monkeypatch):
    path = tmp_path / "m.db"
    conn = db.connect(path)
    lines = [("A (PM)", "Welcome everyone."), ("B (ID)", "What about the price?"),
             ("A (PM)", "Target price is twelve point five euros.")]
    segs = [Segment(segment_id("M1", i), "M1", i, s, i * 10.0, i * 10.0 + 5, t) for i, (s, t) in enumerate(lines)]
    db.add_meeting(conn, "M1", "M1", "test", None, segs)
    db.add_item(conn, "M1", "decision", "Target price is 12.5 euros", "M1-s0002", owner="A (PM)")
    conn.close()
    monkeypatch.setenv("MM_DB", str(path))
    return path


def fake_llm(system, user):
    if "search a meeting transcript" in system:
        return json.dumps({"terms": []})
    return json.dumps({"found": True, "answer": "The target price is 12.5 euros [S1]."})


def run_app():
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    return at


def texts(at):
    return " ".join(str(m.value) for m in at.markdown)


def test_app_loads_and_lists_meetings(db_file):
    at = run_app()
    assert any("M1" in c.value for c in at.caption)
    assert len(at.button) >= 3  # the example questions


def test_asking_a_question_shows_cited_answer(db_file):
    with mock.patch("extract.llm.make_llm", return_value=fake_llm):
        at = run_app()
        at.chat_input[0].set_value("What is the target price?").run()
    assert not at.exception, at.exception
    assert "12.5 euros [S1]" in texts(at)
    assert any("Sources" in e.label for e in at.expander)


def test_unrelated_question_is_not_found_and_model_not_called(db_file):
    called = []

    def spy(system, user):
        called.append(1)
        return fake_llm(system, user)

    with mock.patch("extract.llm.make_llm", return_value=spy):
        at = run_app()
        at.chat_input[0].set_value("What is the capital of France?").run()
    assert "Not found in these meetings." in texts(at) and called == []


def test_missing_key_is_shown_as_a_message_not_a_crash(db_file, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    at = run_app()
    at.chat_input[0].set_value("What is the target price?").run()
    assert not at.exception, at.exception
    assert any("GROQ_API_KEY" in str(w.value) for w in at.warning)


def test_items_page(db_file):
    at = run_app()
    at.sidebar.radio[0].set_value("Decisions & actions").run()
    assert not at.exception, at.exception
    assert len(at.dataframe) == 1


def test_empty_database_shows_instructions(tmp_path, monkeypatch):
    monkeypatch.setenv("MM_DB", str(tmp_path / "empty.db"))
    at = run_app()
    assert any("No meetings are loaded" in i.value for i in at.info)


def test_broad_question_shows_the_helpful_message(db_file):
    called = []

    def spy(system, user):
        called.append(1)
        return fake_llm(system, user)

    with mock.patch("extract.llm.make_llm", return_value=spy):
        at = run_app()
        at.chat_input[0].set_value("What was the meeting about?").run()
    assert "too general" in texts(at) and called == []
