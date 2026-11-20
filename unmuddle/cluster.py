"""Step 2 of the pipeline: group the wrong answers to one question by shared idea.

We use agglomerative clustering with cosine distance. It does not need the number of
groups in advance. We try every possible number of groups and keep the one with the best
silhouette score (how tight each group is compared with its neighbours).

Try it on the demo set:
    python -m unmuddle.cluster --sample ai --embedder minilm
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score

from unmuddle.data import Answer, group_by_question, load_sample
from unmuddle.embed import Embedder, get_embedder


@dataclass
class Cluster:
    question_id: str
    cluster_id: int
    answers: list[Answer]
    # Up to 3 answers closest to the centre of the group: the clearest examples.
    examples: list[Answer] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.answers)


def choose_labels(vectors: np.ndarray, max_groups: int | None = None) -> np.ndarray:
    """Cluster vectors, picking the number of groups with the best silhouette score."""
    n = len(vectors)
    if n == 0:
        return np.array([], dtype=int)
    if n <= 2:
        return np.zeros(n, dtype=int)

    upper = min(max_groups or n - 1, n - 1)
    best_labels = np.zeros(n, dtype=int)
    best_score = -1.0
    for k in range(2, upper + 1):
        model = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
        labels = model.fit_predict(vectors)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(vectors, labels, metric="cosine")
        if score > best_score:
            best_score, best_labels = score, labels
    return best_labels


def _examples(vectors: np.ndarray, members: list[int], answers: list[Answer], top: int = 3) -> list[Answer]:
    centre = vectors[members].mean(axis=0)
    order = sorted(members, key=lambda i: -float(vectors[i] @ centre))
    return [answers[i] for i in order[:top]]


def cluster_wrong_answers(
    question_id: str,
    answers: list[Answer],
    embedder: Embedder,
    max_groups: int | None = None,
) -> list[Cluster]:
    """Cluster the wrong answers to one question. Biggest group first."""
    wrong = [a for a in answers if a.is_correct is False and a.text]
    if not wrong:
        return []
    vectors = embedder.embed([a.text for a in wrong])
    if vectors.shape[1] == 1 and not vectors.any():
        labels = np.zeros(len(wrong), dtype=int)  # nothing to tell apart
    else:
        labels = choose_labels(vectors, max_groups)

    clusters = []
    for label in sorted(set(labels)):
        members = [i for i, lab in enumerate(labels) if lab == label]
        clusters.append(
            Cluster(
                question_id=question_id,
                cluster_id=0,
                answers=[wrong[i] for i in members],
                examples=_examples(vectors, members, wrong),
            )
        )
    clusters.sort(key=lambda c: -c.size)
    for i, c in enumerate(clusters):
        c.cluster_id = i
    return clusters


def cluster_all(answers: list[Answer], embedder: Embedder, **kwargs) -> dict[str, list[Cluster]]:
    return {
        qid: cluster_wrong_answers(qid, group, embedder, **kwargs)
        for qid, group in group_by_question(answers).items()
    }


def grouping_score(clusters: list[Cluster]) -> float | None:
    """Adjusted Rand Index between our groups and the hand myth labels.

    1.0 = perfect match, 0.0 = no better than random. None if there are no labels.
    """
    truth, pred = [], []
    for c in clusters:
        for a in c.answers:
            if a.myth_label:
                truth.append(a.myth_label)
                pred.append(c.cluster_id)
    if len(truth) < 2:
        return None
    return adjusted_rand_score(truth, pred)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster wrong answers in a demo set.")
    parser.add_argument("--sample", choices=["ai", "sql"], default="ai")
    parser.add_argument("--embedder", choices=["minilm", "tfidf"], default="minilm")
    args = parser.parse_args()

    # Uses the hand labels for right/wrong, so this step can be checked on its own.
    _, answers = load_sample(args.sample)
    embedder = get_embedder(args.embedder)
    scores = []
    for qid, clusters in cluster_all(answers, embedder).items():
        score = grouping_score(clusters)
        if score is not None:
            scores.append(score)
        label = "n/a" if score is None else f"{score:.2f}"
        print(f"\n=== {qid}: {len(clusters)} groups (ARI vs hand labels: {label})")
        for c in clusters:
            print(f"  Group {c.cluster_id} ({c.size} students)")
            for a in c.answers:
                print(f"    - {a.text}   [{a.myth_label or '?'}]")
    if scores:
        print(f"\nMean ARI ({args.embedder}): {np.mean(scores):.2f}")


if __name__ == "__main__":
    main()
