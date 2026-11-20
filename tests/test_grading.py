import json
import re

from unmuddle.data import Answer, load_sample
from unmuddle.grading import agreement, grade_all, grade_question
from unmuddle.llm import FakeLLM


def _answers_in(prompt: str) -> list[dict]:
    """Pull the JSON list of answers back out of the prompt."""
    block = prompt.split("Student answers (JSON list of {\"id\", \"answer\"}):")[1]
    return json.loads(re.search(r"\[.*?\]\n", block, re.S).group(0))


def keyword_grader(prompt: str):
    """Fake LLM: an answer is correct if it mentions 'predict'."""
    return [{"id": a["id"], "correct": "predict" in a["answer"].lower()} for a in _answers_in(prompt)]


def test_grade_question_sets_labels():
    questions, answers = load_sample("ai")
    ai1 = [a for a in answers if a.question_id == "ai1"]
    for a in ai1:
        a.is_correct = None
    grade_question(questions["ai1"], ai1, FakeLLM(keyword_grader))
    assert all(a.is_correct is not None for a in ai1)
    assert ai1[0].is_correct is True   # "It predicts the next word..."
    assert ai1[1].is_correct is False  # "It searches Google..."


def test_batches_split_calls():
    questions, answers = load_sample("ai")
    ai1 = [a for a in answers if a.question_id == "ai1"]
    llm = FakeLLM(keyword_grader)
    grade_question(questions["ai1"], ai1, llm, batch_size=5)
    assert len(llm.prompts) == 3  # 12 answers -> 5 + 5 + 2


def test_missing_verdict_counts_as_wrong():
    questions, _ = load_sample("ai")
    answers = [Answer("x", "ai1", "something")]
    grade_question(questions["ai1"], answers, FakeLLM(lambda p: []))
    assert answers[0].is_correct is False


def test_overwrite_false_keeps_existing_labels():
    questions, _ = load_sample("ai")
    answers = [Answer("x", "ai1", "it predicts", is_correct=False), Answer("y", "ai1", "it predicts")]
    llm = FakeLLM(keyword_grader)
    grade_question(questions["ai1"], answers, llm, overwrite=False)
    assert answers[0].is_correct is False  # untouched
    assert answers[1].is_correct is True


def test_grade_all_covers_every_question():
    questions, answers = load_sample("ai")
    llm = FakeLLM(keyword_grader)
    grade_all(questions, answers, llm)
    assert len(llm.prompts) == 3


def test_agreement():
    assert agreement([True, False, True], [True, True, True]) == 2 / 3
    assert agreement([], []) == 0.0
