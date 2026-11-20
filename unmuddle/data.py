"""Load questions and student answers."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from unmuddle.config import SAMPLE_AI_DIR, SAMPLE_DIR


@dataclass
class Question:
    question_id: str
    topic: str
    text: str
    reference_answer: str
    # Misconceptions we already expect for this question (optional).
    known_misconceptions: list[str] = field(default_factory=list)


@dataclass
class Answer:
    student_id: str
    question_id: str
    text: str
    # None means "not graded yet".
    is_correct: bool | None = None
    # Hand label of the misconception behind a wrong answer (only used for evaluation).
    myth_label: str | None = None


def load_questions(path: str | Path) -> dict[str, Question]:
    """Read a questions.json file into a dict keyed by question_id."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    questions = {}
    for item in raw:
        q = Question(
            question_id=item["question_id"],
            topic=item.get("topic", ""),
            text=item["text"],
            reference_answer=item["reference_answer"],
            known_misconceptions=item.get("known_misconceptions", []),
        )
        questions[q.question_id] = q
    return questions


def load_answers(path: str | Path) -> list[Answer]:
    """Read an answers.csv file. The is_correct column is optional."""
    df = pd.read_csv(path, dtype={"student_id": str, "question_id": str})
    has_label = "is_correct" in df.columns
    has_myth = "myth_label" in df.columns
    answers = []
    for row in df.itertuples(index=False):
        label = None
        if has_label and not pd.isna(row.is_correct):
            label = bool(int(row.is_correct))
        myth = None
        if has_myth and not pd.isna(row.myth_label) and str(row.myth_label).strip():
            myth = str(row.myth_label).strip()
        text = "" if pd.isna(row.answer) else str(row.answer).strip()
        answers.append(Answer(row.student_id, row.question_id, text, label, myth))
    return answers


def load_sample(name: str = "ai") -> tuple[dict[str, Question], list[Answer]]:
    """Load a bundled demo set: 'ai' (AI literacy) or 'sql' (intro databases)."""
    folder = {"ai": SAMPLE_AI_DIR, "sql": SAMPLE_DIR}[name]
    return load_questions(folder / "questions.json"), load_answers(folder / "answers.csv")


def group_by_question(answers: list[Answer]) -> dict[str, list[Answer]]:
    """Group answers by question_id, keeping their original order."""
    groups: dict[str, list[Answer]] = defaultdict(list)
    for a in answers:
        groups[a.question_id].append(a)
    return dict(groups)
