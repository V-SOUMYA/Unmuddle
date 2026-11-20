# WhyWrong

**Find out *why* students get it wrong, not just *who* got it wrong.**

WhyWrong reads every student's answer to a question, groups the wrong answers by the
mistake behind them, and gives the teacher a short report like:

> **14 students** think a `LEFT JOIN` removes rows that have no match.
> **2-minute fix:** draw both tables on the board, then show that every row on the left side stays, and the missing values become `NULL`.

## The problem

A teacher or TA with 50+ students cannot read every answer closely enough to see the
shared mistakes. Grading tells you *who* is wrong. It does not tell you *why*. So the same
confusion stays in the class until the exam.

WhyWrong turns a pile of answers into a short list of misconceptions, ranked by how many
students have each one, with a quick way to fix each in class.

## How it works

1. **Check:** split answers into correct and wrong.
2. **Embed:** turn each wrong answer into a vector that captures its meaning.
3. **Cluster:** group wrong answers that share the same idea.
4. **Explain:** an LLM names the misconception behind each group and writes a 2-minute fix.
5. **Report:** the teacher sees the groups, sorted by size, with example answers.

## Evaluation

We measure how well WhyWrong finds the right misconception using the public
[Eedi "Mining Misconceptions in Mathematics"](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics)
dataset, where wrong answers are labeled by experts with the misconception behind them.

## Project layout

```
whywrong/          Python package (the pipeline)
data/sample/       Small demo set: SQL questions + student answers
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
- [ ] Data loading and correctness check
- [ ] Embeddings and clustering of wrong answers
- [ ] LLM labels for each misconception + 2-minute fix
- [ ] Evaluation on the Eedi dataset
- [ ] API, teacher dashboard and live demo
