# Data

## `sample/` (committed)

A small demo set written for this project, in the style of an intro databases course.

- `questions.json`: 3 SQL questions, each with a reference answer.
- `answers.csv`: short student explanations, one row per student per question.
  The `is_correct` column is a hand label, used only to test the pipeline.
  These answers are written for the demo. They are not real student data.

## `eedi/` (not committed)

Used for evaluation. Download it from Kaggle:
<https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/data>

Put these two files in `data/eedi/`:

- `train.csv`
- `misconception_mapping.csv`
