"""Step 1 of the pipeline: decide which answers are correct.

The LLM compares each answer with the reference answer and judges the idea, not the wording.
Answers are sent in batches to keep the number of API calls small.

Try it on the demo set (needs GEMINI_API_KEY):
    python -m unmuddle.grading --sample ai
"""

from __future__ import annotations

import argparse
import json

from unmuddle.data import Answer, Question, group_by_question, load_sample
from unmuddle.llm import LLM

GRADE_PROMPT = """You are grading short student answers.

Question: {question}
Reference answer: {reference}

Rules:
- Judge the idea, not spelling, grammar or wording.
- An answer is correct only if its main idea matches the reference answer.
- If an answer contains a clear wrong idea, mark it incorrect even if part of it is right.
- An empty or "I don't know" answer is incorrect.

Student answers (JSON list of {{"id", "answer"}}):
{answers}

Reply with a JSON list, one item per answer, in the same order:
[{{"id": "<id>", "correct": true or false}}]
"""


def _batches(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def grade_question(
    question: Question,
    answers: list[Answer],
    llm: LLM,
    batch_size: int = 25,
    overwrite: bool = True,
) -> list[Answer]:
    """Set is_correct on each answer to one question. Returns the same list.

    With overwrite=False, answers that already have a label are left alone.
    """
    todo = [a for a in answers if overwrite or a.is_correct is None]
    for batch in _batches(todo, batch_size):
        payload = [{"id": str(i), "answer": a.text} for i, a in enumerate(batch)]
        prompt = GRADE_PROMPT.format(
            question=question.text,
            reference=question.reference_answer,
            answers=json.dumps(payload, ensure_ascii=False),
        )
        reply = llm.json(prompt)
        verdicts = {str(item["id"]): bool(item["correct"]) for item in reply}
        for i, a in enumerate(batch):
            # If the LLM skipped an answer, treat it as wrong so it still gets looked at.
            a.is_correct = verdicts.get(str(i), False)
    return answers


def grade_all(
    questions: dict[str, Question], answers: list[Answer], llm: LLM, **kwargs
) -> list[Answer]:
    """Grade every answer, question by question."""
    for qid, group in group_by_question(answers).items():
        grade_question(questions[qid], group, llm, **kwargs)
    return answers


def agreement(predicted: list[bool], truth: list[bool]) -> float:
    """Share of answers where the grader agrees with the hand label."""
    if not truth:
        return 0.0
    return sum(p == t for p, t in zip(predicted, truth)) / len(truth)


def main() -> None:
    from unmuddle.llm import GeminiLLM

    parser = argparse.ArgumentParser(description="Grade a demo set and compare with hand labels.")
    parser.add_argument("--sample", choices=["ai", "sql"], default="ai")
    args = parser.parse_args()

    questions, answers = load_sample(args.sample)
    truth = [a.is_correct for a in answers]
    grade_all(questions, answers, GeminiLLM())
    predicted = [a.is_correct for a in answers]

    print(f"Graded {len(answers)} answers.")
    print(f"Agreement with hand labels: {agreement(predicted, truth):.1%}")
    for a, t in zip(answers, truth):
        if a.is_correct != t:
            print(f"  disagree [{a.question_id}] {a.text!r}: model={a.is_correct}, label={t}")


if __name__ == "__main__":
    main()
