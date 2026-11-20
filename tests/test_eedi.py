import pandas as pd

from unmuddle.eedi import (
    evaluate,
    grouping_sample,
    load_eedi,
    map_at_k,
    rank_misconceptions,
    recall_at_k,
)
from unmuddle.embed import TfidfEmbedder


def _write_fixture(folder):
    """A tiny dataset in the same format as the Kaggle files."""
    rows = []
    for i in range(6):
        rows.append({
            "QuestionId": i,
            "ConstructName": "Add fractions",
            "SubjectName": "Fractions",
            "CorrectAnswer": "A",
            "QuestionText": f"What is 1/{i + 2} + 1/{i + 2}?",
            "AnswerAText": f"2/{i + 2}",
            "AnswerBText": f"2/{2 * (i + 2)}",
            "AnswerCText": f"1/{i + 2}",
            "AnswerDText": "11",
            "MisconceptionAId": None,
            "MisconceptionBId": 1,   # adds the denominators
            "MisconceptionCId": 2,   # ignores the second fraction
            "MisconceptionDId": None,  # unlabeled wrong answer: skipped
        })
    pd.DataFrame(rows).to_csv(folder / "train.csv", index=False)
    pd.DataFrame({
        "MisconceptionId": [1, 2, 3],
        "MisconceptionName": [
            "Adds the denominators when adding fractions",
            "Ignores the second fraction",
            "Thinks multiplication makes numbers bigger",
        ],
    }).to_csv(folder / "misconception_mapping.csv", index=False)


def test_load_skips_correct_and_unlabeled(tmp_path):
    _write_fixture(tmp_path)
    items, names = load_eedi(tmp_path)
    assert len(items) == 12  # 6 questions x 2 labeled wrong answers
    assert {i.misconception_id for i in items} == {1, 2}
    assert len(names) == 3
    assert "Correct answer: 2/2" in items[0].query()


def test_metrics():
    rankings = [[5, 1, 2], [2, 5, 1], [9, 8, 7]]
    truth = [1, 2, 1]
    assert map_at_k(rankings, truth) == (0.5 + 1.0 + 0.0) / 3
    assert recall_at_k(rankings, truth) == 2 / 3
    assert recall_at_k(rankings, truth, k=1) == 1 / 3
    assert map_at_k([], []) == 0.0


def test_rankings_cover_every_misconception(tmp_path):
    _write_fixture(tmp_path)
    items, names = load_eedi(tmp_path)
    rankings = rank_misconceptions(items, names, TfidfEmbedder())
    assert all(sorted(r) == [1, 2, 3] for r in rankings)


def test_grouping_sample_keeps_common_misconceptions(tmp_path):
    _write_fixture(tmp_path)
    items, _ = load_eedi(tmp_path)
    sample = grouping_sample(items, n_misconceptions=1, min_size=3)
    assert len({i.misconception_id for i in sample}) == 1
    assert len(sample) == 6


def test_evaluate_returns_all_scores(tmp_path):
    _write_fixture(tmp_path)
    result = evaluate(tmp_path, TfidfEmbedder(), limit=5)
    assert result["n_wrong_answers"] == 5
    for key in ("map@25", "recall@25", "grouping_ari"):
        assert -1.0 <= result[key] <= 1.0
    assert result["recall@25"] == 1.0  # only 3 misconceptions, so all fit in the top 25
