"""Evaluate Unmuddle on the Eedi "Mining Misconceptions in Mathematics" dataset.

Experts labeled each wrong multiple-choice answer with the misconception behind it.
We run two checks:

1. Retrieval: for each wrong answer, rank every known misconception by embedding
   similarity. Scored with MAP@25 (the Kaggle competition metric) and Recall@25.
2. Grouping: cluster wrong answers the way the app does, and compare the groups with
   the expert labels using the Adjusted Rand Index (ARI).

Put train.csv and misconception_mapping.csv in data/eedi/, then:
    python -m unmuddle.eedi --embedder minilm
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from unmuddle.cluster import choose_labels
from unmuddle.config import EEDI_DIR, OUTPUT_DIR
from unmuddle.embed import Embedder, get_embedder

LETTERS = "ABCD"


@dataclass
class WrongAnswer:
    question_id: str
    construct: str
    question: str
    correct: str
    wrong: str
    misconception_id: int

    def query(self) -> str:
        """Everything we know about the mistake, as one text to embed."""
        return (
            f"Topic: {self.construct}. Question: {self.question} "
            f"Correct answer: {self.correct}. Student's wrong answer: {self.wrong}."
        )


def load_eedi(folder: str | Path = EEDI_DIR) -> tuple[list[WrongAnswer], dict[int, str]]:
    folder = Path(folder)
    train = pd.read_csv(folder / "train.csv")
    mapping = pd.read_csv(folder / "misconception_mapping.csv")
    names = dict(zip(mapping["MisconceptionId"].astype(int), mapping["MisconceptionName"].astype(str)))

    items = []
    for row in train.to_dict("records"):
        correct_letter = row["CorrectAnswer"]
        for letter in LETTERS:
            mid = row.get(f"Misconception{letter}Id")
            if letter == correct_letter or pd.isna(mid):
                continue
            items.append(
                WrongAnswer(
                    question_id=str(row["QuestionId"]),
                    construct=str(row.get("ConstructName", "")),
                    question=str(row["QuestionText"]),
                    correct=str(row[f"Answer{correct_letter}Text"]),
                    wrong=str(row[f"Answer{letter}Text"]),
                    misconception_id=int(mid),
                )
            )
    return items, names


def rank_misconceptions(items: list[WrongAnswer], names: dict[int, str], embedder: Embedder) -> list[list[int]]:
    """For each wrong answer, every misconception id ordered from most to least similar."""
    ids = list(names)
    texts = [i.query() for i in items] + [names[m] for m in ids]
    vectors = embedder.embed(texts)  # embedded together so TF-IDF shares one vocabulary
    queries, docs = vectors[: len(items)], vectors[len(items) :]
    scores = queries @ docs.T
    order = np.argsort(-scores, axis=1)
    return [[ids[j] for j in row] for row in order]


def map_at_k(rankings: list[list[int]], truth: list[int], k: int = 25) -> float:
    """Mean average precision at k, with one correct misconception per answer."""
    total = 0.0
    for ranked, t in zip(rankings, truth):
        top = ranked[:k]
        if t in top:
            total += 1.0 / (top.index(t) + 1)
    return total / len(truth) if truth else 0.0


def recall_at_k(rankings: list[list[int]], truth: list[int], k: int = 25) -> float:
    if not truth:
        return 0.0
    return sum(t in r[:k] for r, t in zip(rankings, truth)) / len(truth)


def grouping_sample(items: list[WrongAnswer], n_misconceptions: int = 20, min_size: int = 3, seed: int = 0) -> list[WrongAnswer]:
    """Pick misconceptions that appear at least `min_size` times, and all their wrong answers.

    This mimics a class where several learners share each wrong idea.
    """
    counts = Counter(i.misconception_id for i in items)
    common = sorted(m for m, c in counts.items() if c >= min_size)
    random.Random(seed).shuffle(common)
    chosen = set(common[:n_misconceptions])
    return [i for i in items if i.misconception_id in chosen]


def grouping_ari(sample: list[WrongAnswer], embedder: Embedder) -> float:
    if len(sample) < 3:
        return 0.0
    vectors = embedder.embed([i.query() for i in sample])
    max_groups = min(len(sample) - 1, 3 * len({i.misconception_id for i in sample}))
    labels = choose_labels(vectors, max_groups=max_groups)
    return adjusted_rand_score([i.misconception_id for i in sample], labels)


def evaluate(folder: str | Path, embedder: Embedder, limit: int | None = None, seed: int = 0) -> dict:
    items, names = load_eedi(folder)
    eval_items = items
    if limit and limit < len(items):
        eval_items = random.Random(seed).sample(items, limit)
    rankings = rank_misconceptions(eval_items, names, embedder)
    truth = [i.misconception_id for i in eval_items]
    sample = grouping_sample(items, seed=seed)
    return {
        "embedder": embedder.name,
        "n_wrong_answers": len(eval_items),
        "n_misconceptions": len(names),
        "map@25": round(map_at_k(rankings, truth), 4),
        "recall@25": round(recall_at_k(rankings, truth), 4),
        "grouping_n_answers": len(sample),
        "grouping_n_misconceptions": len({i.misconception_id for i in sample}),
        "grouping_ari": round(grouping_ari(sample, embedder), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate on the Eedi misconception dataset.")
    parser.add_argument("--embedder", choices=["minilm", "gemini", "tfidf"], default="minilm")
    parser.add_argument("--folder", default=str(EEDI_DIR))
    parser.add_argument("--limit", type=int, default=None, help="Only score this many wrong answers.")
    args = parser.parse_args()

    result = evaluate(args.folder, get_embedder(args.embedder), limit=args.limit)
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / f"eedi_{args.embedder}.json"
    path.write_text(json.dumps(result, indent=2))
    for key, value in result.items():
        print(f"{key:28s} {value}")
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
