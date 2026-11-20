import pytest
from fastapi.testclient import TestClient

from unmuddle.api import create_app
from unmuddle.embed import TfidfEmbedder
from unmuddle.llm import FakeLLM

from tests.fakes import fake_respond


@pytest.fixture
def client(tmp_path):
    app = create_app(
        db_path=tmp_path / "test.db",
        llm_factory=lambda: FakeLLM(fake_respond),
        embedder_factory=TfidfEmbedder,
    )
    return TestClient(app)


def test_lessons_hide_reference_answers(client):
    lessons = client.get("/api/lessons").json()
    assert len(lessons) == 12
    assert "reference_answer" not in lessons[0]["questions"][0]


def test_create_session_and_join(client):
    code = client.post("/api/sessions", json={"lesson_id": "L08"}).json()["code"]
    assert len(code) == 6
    s = client.get(f"/api/sessions/{code.lower()}").json()  # codes are case-insensitive
    assert s["title"].startswith("Lesson 8")
    assert s["n_learners"] == 0 and s["analyzed"] is False
    assert client.get(f"/api/sessions/{code}/qr.svg").headers["content-type"].startswith("image/svg")


def test_unknown_lesson_and_session(client):
    assert client.post("/api/sessions", json={"lesson_id": "nope"}).status_code == 404
    assert client.get("/api/sessions/ZZZZZZ").status_code == 404


def test_submission_validation(client):
    code = client.post("/api/sessions", json={"lesson_id": "L08"}).json()["code"]
    url = f"/api/sessions/{code}/responses"
    assert client.post(url, json={"name": "A", "answers": {"bogus": "x"}}).status_code == 400
    assert client.post(url, json={"name": "A", "answers": {"L08-q1": "   "}}).status_code == 400
    assert client.post(url, json={"name": "", "answers": {"L08-q1": "x"}}).status_code == 422
    ok = client.post(url, json={"name": "A", "answers": {"L08-q1": "it predicts", "bogus": "x"}})
    assert ok.status_code == 200
    assert client.get(f"/api/sessions/{code}").json()["n_answers"] == 1


def test_analyze_needs_answers(client):
    code = client.post("/api/sessions", json={"lesson_id": "L08"}).json()["code"]
    assert client.post(f"/api/sessions/{code}/analyze").status_code == 400
    assert client.get(f"/api/sessions/{code}/report").status_code == 404


def test_full_demo_flow(client):
    code = client.post("/api/demo").json()["code"]
    assert client.get(f"/api/sessions/{code}").json()["n_learners"] == 12

    # A new learner joins before analysis.
    sub = client.post(
        f"/api/sessions/{code}/responses",
        json={"name": "Maya", "answers": {"ai1": "It searches Google for the answer", "ai2": "it overfit"}},
    )
    learner = sub.json()["learner_id"]
    before = client.get(f"/api/sessions/{code}/learners/{learner}/feedback").json()
    assert before["ready"] is False

    report = client.post(f"/api/sessions/{code}/analyze").json()
    assert report["n_learners"] == 13
    assert report["new_answers_since_analysis"] is False
    for q in report["questions"]:
        wrong = q["n_answers"] - q["n_correct"]
        assert sum(m["size"] for m in q["misconceptions"]) == wrong

    fb = client.get(f"/api/sessions/{code}/learners/{learner}/feedback").json()
    assert fb["ready"] is True
    by_q = {i["question_id"]: i for i in fb["items"]}
    assert by_q["ai2"]["correct"] is True
    wrong_item = by_q["ai1"]
    assert wrong_item["correct"] is False
    mid = wrong_item["misconception"]["misconception_id"]
    assert wrong_item["misconception"]["follow_up_question"]

    # Follow-up: first attempt wrong, second right. Only the latest counts.
    url = f"/api/sessions/{code}/learners/{learner}/followup"
    assert client.post(url, json={"misconception_id": mid, "answer": "it googles"}).json()["fixed"] is False
    assert client.post(url, json={"misconception_id": mid, "answer": "it predicts words"}).json()["fixed"] is True

    report = client.get(f"/api/sessions/{code}/report").json()
    m = next(m for q in report["questions"] for m in q["misconceptions"] if m["misconception_id"] == mid)
    assert m["follow_up_stats"] == {"attempted": 1, "fixed": 1}

    # A learner cannot answer a follow-up for a misconception they don't hold.
    other = next(
        m["misconception_id"]
        for q in report["questions"] for m in q["misconceptions"] if learner not in m["student_ids"]
    )
    assert client.post(url, json={"misconception_id": other, "answer": "x"}).status_code == 404

    # New answers after analysis are flagged so the leader can re-run.
    client.post(f"/api/sessions/{code}/responses", json={"name": "Late", "answers": {"ai3": "bias"}})
    assert client.get(f"/api/sessions/{code}/report").json()["new_answers_since_analysis"] is True


def test_followup_before_analysis_is_rejected(client):
    code = client.post("/api/demo").json()["code"]
    learner = client.post(
        f"/api/sessions/{code}/responses", json={"name": "A", "answers": {"ai1": "x"}}
    ).json()["learner_id"]
    r = client.post(f"/api/sessions/{code}/learners/{learner}/followup", json={"misconception_id": "ai1-m0", "answer": "x"})
    assert r.status_code == 409
