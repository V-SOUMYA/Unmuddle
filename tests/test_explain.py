from unmuddle.cluster import cluster_wrong_answers
from unmuddle.data import Answer, Question, load_sample
from unmuddle.embed import TfidfEmbedder
from unmuddle.explain import Misconception, check_follow_up, explain_question
from unmuddle.llm import FakeLLM
from unmuddle.pipeline import analyze

from tests.fakes import fake_respond


def _clusters(qid="ai1"):
    questions, answers = load_sample("ai")
    group = [a for a in answers if a.question_id == qid]
    return questions[qid], cluster_wrong_answers(qid, group, TfidfEmbedder())


def test_merging_keeps_every_student_once():
    question, clusters = _clusters()
    ms = explain_question(question, clusters, FakeLLM(fake_respond))
    students = [s for m in ms for s in m.student_ids]
    expected = [a.student_id for c in clusters for a in c.answers]
    assert sorted(students) == sorted(expected)
    assert len(ms) == len(clusters) - 1  # first two groups merged
    assert [m.size for m in ms] == sorted((m.size for m in ms), reverse=True)
    assert [m.misconception_id for m in ms] == [f"ai1-m{i}" for i in range(len(ms))]


def test_skipped_and_bad_group_ids_are_handled():
    question, clusters = _clusters()

    def lazy(prompt):  # names only group 0, repeats it, and adds nonsense ids
        return {"misconceptions": [
            {"group_ids": [0, 0, 99, "x"], "title": "Only one"},
            {"group_ids": [0], "title": "Duplicate"},
        ]}

    ms = explain_question(question, clusters, FakeLLM(lazy))
    assert len(ms) == len(clusters)  # skipped groups still appear
    titles = [m.title for m in ms]
    assert "Only one" in titles and "Duplicate" not in titles
    assert sum(t.startswith("Other wrong idea") for t in titles) == len(clusters) - 1


def test_garbage_reply_still_returns_all_groups():
    question, clusters = _clusters()
    ms = explain_question(question, clusters, FakeLLM(lambda p: "not json"))
    assert len(ms) == len(clusters)


def test_no_clusters_means_no_llm_call():
    question, _ = _clusters()
    llm = FakeLLM(fake_respond)
    assert explain_question(question, [], llm) == []
    assert llm.prompts == []


def test_known_misconceptions_reach_the_prompt():
    q = Question("x", "t", "Q?", "A.", known_misconceptions=["AI searches the web"])
    cl = cluster_wrong_answers("x", [Answer("s1", "x", "it googles", False)], TfidfEmbedder())
    llm = FakeLLM(fake_respond)
    explain_question(q, cl, llm)
    assert "AI searches the web" in llm.prompts[0]


def test_check_follow_up():
    m = Misconception("ai1", "ai1-m0", "Searches the web", follow_up_question="?", follow_up_answer="!")
    assert check_follow_up(m, "it predicts tokens", FakeLLM(fake_respond))["fixed"] is True
    assert check_follow_up(m, "it googles", FakeLLM(fake_respond))["fixed"] is False
    assert check_follow_up(m, "x", FakeLLM(lambda p: None)) == {"fixed": False, "feedback": ""}


def test_full_pipeline_report():
    questions, answers = load_sample("ai")
    report = analyze(questions, answers, FakeLLM(fake_respond), TfidfEmbedder())
    assert len(report.questions) == 3
    for qr in report.questions:
        assert qr.n_answers == 12
        wrong = qr.n_answers - qr.n_correct
        assert sum(m.size for m in qr.misconceptions) == wrong

    wrong_student = next(a for a in answers if a.question_id == "ai1" and not a.is_correct)
    assert report.misconception_for(wrong_student.student_id, "ai1") is not None
    assert report.misconception_for("l01", "ai1") is None  # l01 was right
    assert "2-minute fix" in report.to_markdown()
    assert report.to_dict()["questions"][0]["misconceptions"][0]["size"] > 0
