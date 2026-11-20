"""The full pipeline: grade -> cluster -> explain, packaged as one report.

Run it on a demo set (needs GEMINI_API_KEY):
    python -m unmuddle.pipeline --sample ai
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field

from unmuddle.cluster import cluster_wrong_answers
from unmuddle.config import OUTPUT_DIR
from unmuddle.data import Answer, Question, group_by_question, load_sample
from unmuddle.embed import Embedder
from unmuddle.explain import Misconception, explain_question
from unmuddle.grading import grade_question
from unmuddle.llm import LLM


@dataclass
class QuestionReport:
    question: Question
    n_answers: int
    n_correct: int
    misconceptions: list[Misconception] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "question_id": self.question.question_id,
            "topic": self.question.topic,
            "text": self.question.text,
            "reference_answer": self.question.reference_answer,
            "n_answers": self.n_answers,
            "n_correct": self.n_correct,
            "misconceptions": [m.to_dict() for m in self.misconceptions],
        }


@dataclass
class Report:
    questions: list[QuestionReport]

    def to_dict(self) -> dict:
        return {"questions": [q.to_dict() for q in self.questions]}

    def misconception_for(self, student_id: str, question_id: str) -> Misconception | None:
        """The misconception a given learner holds on a question, if any."""
        for q in self.questions:
            if q.question.question_id != question_id:
                continue
            for m in q.misconceptions:
                if student_id in m.student_ids:
                    return m
        return None

    def to_markdown(self) -> str:
        lines = []
        for q in self.questions:
            lines.append(f"## {q.question.topic or q.question.question_id}")
            lines.append(f"{q.n_correct}/{q.n_answers} correct\n")
            for m in q.misconceptions:
                lines.append(f"### {m.size} students: {m.title}")
                lines.append(f"- Why it's wrong: {m.explanation}")
                lines.append(f"- 2-minute fix: {m.fix}")
                lines.append(f"- Check question: {m.follow_up_question}")
                lines.append("")
        return "\n".join(lines)


def analyze(
    questions: dict[str, Question],
    answers: list[Answer],
    llm: LLM,
    embedder: Embedder,
    regrade: bool = True,
) -> Report:
    """Run the whole pipeline. With regrade=False, existing is_correct labels are kept."""
    reports = []
    for qid, group in group_by_question(answers).items():
        question = questions[qid]
        grade_question(question, group, llm, overwrite=regrade)
        clusters = cluster_wrong_answers(qid, group, embedder)
        reports.append(
            QuestionReport(
                question=question,
                n_answers=len(group),
                n_correct=sum(1 for a in group if a.is_correct),
                misconceptions=explain_question(question, clusters, llm),
            )
        )
    return Report(reports)


def main() -> None:
    from unmuddle.embed import get_embedder
    from unmuddle.llm import GeminiLLM

    parser = argparse.ArgumentParser(description="Run the full pipeline on a demo set.")
    parser.add_argument("--sample", choices=["ai", "sql"], default="ai")
    parser.add_argument("--embedder", choices=["minilm", "gemini", "tfidf"], default="minilm")
    args = parser.parse_args()

    questions, answers = load_sample(args.sample)
    report = analyze(questions, answers, GeminiLLM(), get_embedder(args.embedder))

    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / f"report_{args.sample}.json"
    path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    print(report.to_markdown())
    print(f"Saved full report to {path}")


if __name__ == "__main__":
    main()
