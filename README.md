# Unmuddle

**See what your class got wrong, and why.**

Learners answer in their own words on their phones. Unmuddle groups the wrong answers by
the idea behind them, ranks each misconception by how many learners hold it, and gives the
leader a 2-minute way to fix it in class:

> **5 learners** think *"ChatGPT searches the internet for each answer."*
> **2-minute fix:** turn off the Wi-Fi and ask a downloaded model a question. It still
> answers, because it predicts words from what it learned in training.

Leaders pick one of 12 ready-made beginner AI lessons or write their own questions for any
subject. Each learner then gets kind feedback and a new check question. When they get it right,
their dot on the leader's dashboard turns from green to plum.

**Live demo:** add the Render URL here after deploying · **Video:** add the YouTube link

## The problem

A teacher or TA with 50 students can't read every answer closely enough to see the shared
mistakes. Grading tells you *who* is wrong, not *why*, so the same confusion stays in the
class until the exam.

This matters even more now that AI is taught by students to students, in clubs and
chapters like those of the [ML Empowerment Foundation](https://mlempowermentfoundation.org/).
Student leaders aren't trained teachers, and beginners pick up AI myths fast:
"ChatGPT searches the internet", "computers can't be biased", "a bigger model fixes
everything".

## How it works

```
answers ──> grade ──> embed wrong answers ──> cluster ──> explain ──> dashboard
             (LLM)      (sentence vectors)    (by meaning)   (LLM)        │
                                                                          v
                           dot turns plum <── check question <── learner feedback
```

1. **Grade:** Gemini compares each answer with the reference answer, judging the idea,
   not the wording.
2. **Embed:** each wrong answer becomes a sentence embedding that captures its meaning.
3. **Cluster:** agglomerative clustering on cosine distance groups answers that share an
   idea. The number of groups is picked automatically with the silhouette score.
4. **Explain:** Gemini merges groups that are the same misconception, names each one as
   the belief students hold, and writes why it's wrong, why it's tempting, a 2-minute
   fix, learner feedback and a new check question.
5. **Close the loop:** learners answer the check question, and the dashboard shows how
   many fixed each misconception.

## Results

**Grouping on the AI demo class** (wrong answers hand-labeled with their true myth;
Adjusted Rand Index, 1.0 = perfect, 0 = random):

| Embedder | Mean ARI |
|---|---|
| TF-IDF (word overlap, baseline) | 0.29 |
| MiniLM (meaning) | _run `python -m unmuddle.cluster --embedder minilm`_ |
| Gemini embeddings | _run `python -m unmuddle.cluster --embedder gemini`_ |

Word overlap fails because learners say the same wrong idea in different words:
"browses the web" and "searches Google" share no words but mean the same thing.

**Eedi dataset** (experts labeled real students' wrong maths answers with one of 2,500+
misconceptions):

| Embedder | MAP@25 | Recall@25 | Grouping ARI |
|---|---|---|---|
| TF-IDF (baseline) | _run it_ | _run it_ | _run it_ |
| MiniLM | _run it_ | _run it_ | _run it_ |

**Tests:** 41 automated tests cover the pipeline, the API and the edge cases (`pytest`).

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then add your Gemini API key
uvicorn unmuddle.api:app --reload
```

Open http://localhost:8000 and click **Try a demo class**.

## Tools

| Command | What it does |
|---|---|
| `python -m unmuddle.grading --sample ai` | Grades the demo answers; prints agreement with hand labels |
| `python -m unmuddle.cluster --sample ai --embedder minilm` | Prints the groups of wrong answers and their ARI |
| `python -m unmuddle.pipeline --sample ai` | Full report; saved to `outputs/report_ai.json` |
| `python -m unmuddle.eedi --embedder minilm` | Eedi evaluation (put the Kaggle files in `data/eedi/` first) |
| `pytest` | Runs all tests |

`--embedder` takes `minilm` (local), `gemini` (hosted) or `tfidf` (baseline).
`--sample` takes `ai` or `sql`.

## API

| Step | Endpoint |
|---|---|
| Leader picks a lesson | `POST /api/sessions` -> 6-letter code |
| ...or writes their own questions (1-5) | `POST /api/sessions/custom` -> 6-letter code |
| Learners scan the QR code | `GET /api/sessions/{code}/qr.svg` |
| Learners answer | `POST /api/sessions/{code}/responses` |
| Leader runs the analysis | `POST /api/sessions/{code}/analyze` |
| Learner sees feedback | `GET /api/sessions/{code}/learners/{id}/feedback` |
| Learner answers the check question | `POST /api/sessions/{code}/learners/{id}/followup` |
| Dashboard shows "fixed" counts | `GET /api/sessions/{code}/report` |
| Demo class with 12 learners | `POST /api/demo` |

Interactive docs at `/docs`. Settings (environment variables): `GEMINI_API_KEY`,
`UNMUDDLE_EMBEDDER` (`minilm`, `gemini` or `tfidf`), `UNMUDDLE_DB` (SQLite file path).

## Deploy

The live site uses Gemini for embeddings, so it needs no PyTorch and fits a free server.

1. On [Render](https://render.com), choose **New > Blueprint** and pick this repo.
2. Paste your `GEMINI_API_KEY` when asked.
3. Render builds the `Dockerfile` and gives you a public URL.

Sessions are kept in SQLite on the server's temporary disk, which is fine for a demo.

## Project layout

```
unmuddle/
  data.py        Load questions, answers and lessons
  llm.py         Gemini wrapper that returns JSON (plus a fake for tests)
  grading.py     Step 1: right or wrong
  embed.py       Sentence embeddings: MiniLM, Gemini, TF-IDF
  cluster.py     Step 2: group wrong answers by meaning
  explain.py     Step 3: name misconceptions, fixes, feedback, check questions
  pipeline.py    Steps 1-3 as one report
  eedi.py        Evaluation on the Eedi dataset
  store.py       SQLite storage for sessions
  api.py         FastAPI app
web/             Landing page, question writer, learner page, leader dashboard
data/lessons.json  Question bank: 12 beginner AI lessons, 24 questions with known myths
data/sample_ai/  Demo class: AI-literacy answers with hand labels
data/sample/     Second demo set: intro SQL
docs/            Devpost write-up and demo video script
tests/           41 tests
```
