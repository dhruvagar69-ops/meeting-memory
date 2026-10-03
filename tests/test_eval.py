import json

from ask.ask import answer_question
from eval.run_eval import load_questions, run, score_correct, summarize, write_csv
from tests.test_ask import make_conn


def llm_for(answer):
    def llm(system, user):
        if "search a meeting transcript" in system:
            return json.dumps({"terms": ["euros", "cost"]})
        return json.dumps(answer)
    return llm


QUESTIONS = [
    {"id": "q1", "question": "What is the target price?", "gold_segments": ["M1-s0002"], "multi_meeting": False},
    {"id": "q2", "question": "Which meetings mention buttons or price?", "gold_segments": ["M2-s0000"], "multi_meeting": True},
    {"id": "q3", "question": "Who is the CEO of the company?", "answerable": False, "gold_segments": []},
]


def test_run_and_summarize():
    rows = run(make_conn(), [dict(q, answerable=q.get("answerable", True)) for q in QUESTIONS],
               llm_for({"found": True, "answer": "Twelve and a half euros [S1]."}))
    q1, q2, q3 = rows
    assert q1["retrieval_hit"] is True and q1["citation_hit"] is True and q1["found"] and q1["verified"]
    assert q3["found"] is False and q3["retrieval_hit"] == ""   # nothing to retrieve for "CEO company"
    s = summarize(rows)
    assert s["answerable"] == 2 and s["unanswerable_correctly_not_found"] == "1/1"


def test_expansion_finds_what_plain_keywords_miss():
    conn = make_conn()
    plain = answer_question(conn, "How many euros was it?", llm_for({"found": True, "answer": "x [S1]"}))
    assert plain.found  # 'euros' appears in the transcript, so even plain search works here
    q = [{"id": "q", "question": "What does it cost?", "gold_segments": ["M1-s0002"], "answerable": True,
          "multi_meeting": False}]
    without = run(conn, q, llm_for({"found": True, "answer": "x [S1]"}))[0]
    with_exp = run(conn, q, llm_for({"found": True, "answer": "x [S1]"}), expand=True)[0]
    assert without["retrieval_hit"] is False and with_exp["retrieval_hit"] is True


def test_csv_and_scoring(tmp_path):
    rows = run(make_conn(), [dict(QUESTIONS[0], answerable=True)], llm_for({"found": True, "answer": "ok [S1]"}))
    path = tmp_path / "r.csv"
    write_csv(rows, path)
    text = path.read_text(encoding="utf-8-sig").replace(",,\n", ",yes,\n", 1)  # fill `correct`
    path.write_text(text, encoding="utf-8-sig")
    assert score_correct(path)["unlabelled"] in (0, 1)


def test_load_questions_defaults(tmp_path):
    f = tmp_path / "q.json"
    f.write_text(json.dumps([{"id": "a", "question": "x"}]))
    q = load_questions(f)[0]
    assert q["answerable"] is True and q["gold_segments"] == [] and q["multi_meeting"] is False


def test_skip_and_on_row_and_csv_resume(tmp_path):
    from eval.run_eval import read_csv_rows
    seen = []
    qs = [dict(QUESTIONS[0], answerable=True), dict(QUESTIONS[1], answerable=True)]
    rows = run(make_conn(), qs, llm_for({"found": True, "answer": "ok [S1]"}), skip_ids={"q1"}, on_row=seen.append)
    assert [r["id"] for r in rows] == ["q2"] and [r["id"] for r in seen] == ["q2"]
    path = tmp_path / "r.csv"
    write_csv(rows, path)
    back = read_csv_rows(path)
    assert back[0]["id"] == "q2" and back[0]["found"] is True and back[0]["answerable"] is True
