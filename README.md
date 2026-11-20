# Unmuddle

**Find out *why* students get it wrong, not just *who* got it wrong.**

Unmuddle reads every student's answer to a question, groups the wrong answers by the
mistake behind them, and gives the teacher a short report like:

> **14 students** think a `LEFT JOIN` removes rows that have no match.
> **2-minute fix:** draw both tables on the board, then show that every row on the left side stays, and the missing values become `NULL`.

## The problem

A teacher or TA with 50+ students cannot read every answer closely enough to see the
shared mistakes. Grading tells you *who* is wrong. It does not tell you *why*. So the same
confusion stays in the class until the exam.

Unmuddle turns a pile of answers into a short list of misconceptions, ranked by how many
students have each one, with a quick way to fix each in class.

## Built for student-led AI education

AI is now taught by students to students, in clubs and chapters like those of the
[ML Empowerment Foundation](https://mlempowermentfoundation.org/). Student leaders are
not trained teachers, and beginners pick up AI myths fast:
*"ChatGPT searches the internet"*, *"computers can't be biased"*,
*"a bigger model fixes everything"*.

Unmuddle helps a chapter leader see these myths in their group after a workshop, and
gives them a simple way to fix each one. It works for any subject: the repo includes
an AI-literacy demo set and an intro SQL demo set.

## How it works

1. **Check:** split answers into correct and wrong.
2. **Embed:** turn each wrong answer into a vector that captures its meaning.
3. **Cluster:** group wrong answers that share the same idea.
4. **Explain:** an LLM names the misconception behind each group and writes a 2-minute fix.
5. **Report:** the teacher sees the groups, sorted by size, with example answers.

## Evaluation

We measure how well Unmuddle finds the right misconception using the public
[Eedi "Mining Misconceptions in Mathematics"](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics)
dataset, where wrong answers are labeled by experts with the misconception behind them.

## Project layout

```
unmuddle/          Python package (the pipeline)
data/sample_ai/    Main demo set: AI-literacy questions + learner answers
data/sample/       Second demo set: SQL questions + student answers
data/eedi/         Eedi dataset goes here (not committed, see data/README.md)
tests/             Unit tests
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Roadmap

- [x] Project setup and sample data
- [x] AI-literacy demo set
- [x] Data loading and right/wrong grading
- [x] Embeddings and clustering of wrong answers
- [x] Name each misconception, 2-minute fix, personal feedback and follow-up question
- [ ] Evaluation on the Eedi dataset
- [ ] API and live sessions
- [ ] Learner page (join by QR code) and live leader dashboard
- [ ] Question bank for a 12-lesson beginner AI curriculum
- [ ] Live demo

## Run the tests

```bash
pytest
```

## Try the grader

Needs a Gemini API key in `.env`.

```bash
python -m unmuddle.grading --sample ai
```

It grades the demo answers and shows how often it agrees with the hand labels.

## Try the clustering

```bash
python -m unmuddle.cluster --sample ai --embedder minilm   # meaning-based (main)
python -m unmuddle.cluster --sample ai --embedder tfidf    # word-overlap baseline
```

It prints the groups of wrong answers for each question and scores them against hand
labels with the Adjusted Rand Index (ARI: 1.0 = perfect, 0 = random).

| Embedder | Mean ARI on AI demo set |
|---|---|
| TF-IDF (word overlap, baseline) | 0.29 |
| MiniLM (meaning) | _run it and add your number_ |

Word overlap fails because students say the same wrong idea in different words:
"browses the web" and "searches Google" share no words but mean the same thing.

## Run the full pipeline

Needs a Gemini API key in `.env`.

```bash
python -m unmuddle.pipeline --sample ai
```

For each question you get every misconception, biggest first, with:

- **title**: the wrong belief, in the students' own terms
- **explanation** and **why it's tempting**
- **2-minute fix**: a quick class activity or analogy
- **learner feedback**: a kind note for each learner who holds the belief
- **follow-up question**: a new question that checks whether the belief is gone

The LLM also merges groups that turn out to be the same misconception. The full report
is saved to `outputs/report_ai.json`.
