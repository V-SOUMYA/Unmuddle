import numpy as np

from unmuddle.cluster import choose_labels, cluster_all, cluster_wrong_answers, grouping_score
from unmuddle.data import Answer, load_sample
from unmuddle.embed import TfidfEmbedder


class FixedEmbedder:
    """Returns hand-made vectors, so tests do not depend on a model."""

    name = "fixed"

    def __init__(self, table: dict[str, list[float]]):
        self.table = table

    def embed(self, texts):
        v = np.array([self.table[t] for t in texts], dtype=float)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def _wrong(text, myth=None):
    return Answer("s", "q", text, is_correct=False, myth_label=myth)


def test_choose_labels_finds_two_clear_groups():
    rng = np.random.default_rng(0)
    a = np.array([1.0, 0, 0]) + rng.normal(0, 0.05, (5, 3))
    b = np.array([0, 1.0, 0]) + rng.normal(0, 0.05, (5, 3))
    labels = choose_labels(np.vstack([a, b]))
    assert len(set(labels[:5])) == 1
    assert len(set(labels[5:])) == 1
    assert labels[0] != labels[5]


def test_small_inputs():
    assert len(choose_labels(np.zeros((0, 3)))) == 0
    assert list(choose_labels(np.ones((2, 3)))) == [0, 0]


def test_groups_match_known_myths_and_sort_by_size():
    table = {
        "searches google": [1, 0, 0],
        "browses the web": [0.95, 0.05, 0],
        "goes online": [0.9, 0.1, 0],
        "thinks like a human": [0, 1, 0],
        "reasons like a brain": [0.05, 0.95, 0],
    }
    answers = [_wrong(t, "web" if i < 3 else "human") for i, t in enumerate(table)]
    answers.append(Answer("s", "q", "predicts next token", is_correct=True))

    clusters = cluster_wrong_answers("q", answers, FixedEmbedder(table))

    assert [c.size for c in clusters] == [3, 2]
    assert [c.cluster_id for c in clusters] == [0, 1]
    assert grouping_score(clusters) == 1.0
    # The correct answer is never clustered.
    assert all(a.is_correct is False for c in clusters for a in c.answers)


def test_examples_are_members_and_at_most_three():
    _, answers = load_sample("ai")
    for clusters in cluster_all(answers, TfidfEmbedder()).values():
        for c in clusters:
            assert 1 <= len(c.examples) <= 3
            assert all(e in c.answers for e in c.examples)


def test_every_wrong_answer_is_in_exactly_one_group():
    _, answers = load_sample("ai")
    groups = cluster_all(answers, TfidfEmbedder())
    clustered = [id(a) for cs in groups.values() for c in cs for a in c.answers]
    wrong = [id(a) for a in answers if a.is_correct is False]
    assert sorted(clustered) == sorted(wrong)


def test_no_wrong_answers_gives_no_groups():
    answers = [Answer("s", "q", "right", is_correct=True)]
    assert cluster_wrong_answers("q", answers, TfidfEmbedder()) == []


def test_grouping_score_without_labels_is_none():
    clusters = cluster_wrong_answers("q", [_wrong("a b"), _wrong("c d")], TfidfEmbedder())
    assert grouping_score(clusters) is None


def test_myth_labels_load_for_ai_sample():
    _, answers = load_sample("ai")
    wrong = [a for a in answers if a.is_correct is False]
    assert all(a.myth_label for a in wrong)
    assert all(a.myth_label is None for a in answers if a.is_correct)
