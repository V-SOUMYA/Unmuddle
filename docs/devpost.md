# Devpost submission: Unmuddle

## Project title

Unmuddle: see what your class got wrong, and why

## Tagline

Learners answer in their own words. Unmuddle groups the wrong answers by the idea behind
them and gives the leader a 2-minute fix for each one.

## Problem statement

AI is now taught by students to students, in clubs and chapters around the world. A
student leader running a workshop can't read 30 or 50 free-text answers closely enough
to see the shared mistakes. Grading tells you *who* is wrong. It doesn't tell you *why*.

So beginners leave with myths that stick: "ChatGPT searches the internet", "computers
can't be biased", "a bigger model fixes overfitting". These myths shape how they use and
judge AI long after the workshop ends.

## Solution overview

The leader picks a lesson and shows a QR code. Learners join on their phones and answer
in their own words. With one press, Unmuddle:

1. Checks each answer against the lesson's reference answer, judging the idea and not
   the wording.
2. Turns every wrong answer into a sentence embedding and clusters answers that share an
   idea.
3. Uses an LLM to name each misconception as the belief students hold, merge duplicate
   groups, explain why it's wrong and why it's tempting, and write a 2-minute fix the
   leader can use right away.
4. Sends each learner kind, personal feedback and a new check question.
5. Updates the dashboard live: one dot per learner, and each dot turns from grey to indigo
   when that learner gets the check question right.

The leader goes from "half the room got it wrong" to "5 people think ChatGPT searches
the web, here's how to fix that in 2 minutes", and can see the fix working.

## Key features

- **Misconception dashboard:** wrong ideas ranked by how many learners hold them, with
  example answers.
- **2-minute fixes:** a quick demo, analogy or activity for each misconception.
- **Personal feedback:** each learner sees what they got right, the idea behind their
  mix-up, and a new question to try.
- **Closing the loop:** follow-up results flow back to the dashboard, so the leader sees
  which fixes worked.
- **Ready-made question bank:** 12 beginner AI lessons (what AI is, data, overfitting,
  neural networks, LLMs, multimodal AI, prompt engineering, bias, the future of AI) with
  questions, reference answers and the myths we expect.
- **Join in seconds:** a 6-letter code and a QR code, with no accounts or app installs.
- **Try it instantly:** a demo class with 12 learners' answers already filled in.
- **Works for any subject:** the repo includes a second demo set for intro SQL.

## How we measured it

- **Grouping on our demo class:** wrong answers are hand-labeled with their true myth,
  and we score how well the groups match using the Adjusted Rand Index.
  The word-overlap baseline (TF-IDF) scores 0.29; meaning-based embeddings score higher.
- **Expert-labeled data:** we evaluate on the Eedi "Mining Misconceptions in Mathematics"
  dataset, where experts labeled real students' wrong answers with one of 2,500+
  misconceptions. We report MAP@25 and Recall@25 for finding the right misconception, and
  ARI for grouping.
- **Grader agreement:** how often the AI grader agrees with hand labels on the demo set.
- **Tests:** 39 automated tests cover the pipeline, the API and every edge case we hit.

Before submitting, put your own numbers into this section from `python -m unmuddle.cluster`,
`python -m unmuddle.eedi` and `python -m unmuddle.grading`.

## Technologies used

- Python, FastAPI, SQLite
- Google Gemini (grading, explanations, follow-up checks) through the google-genai SDK
- Sentence embeddings: Gemini embeddings on the live site, MiniLM
  (sentence-transformers) locally, TF-IDF as a baseline
- scikit-learn: agglomerative clustering with cosine distance, silhouette-based choice of
  the number of groups, Adjusted Rand Index
- Vanilla HTML, CSS and JavaScript front end, segno for QR codes
- pytest, Docker, Render

## Target users

- **Student leaders** of AI clubs and chapters, like those of the ML Empowerment
  Foundation, who teach beginners but aren't trained teachers.
- **Teachers and TAs** with large classes who want to know what to re-teach.
- **Learners**, who get kind, specific feedback instead of just "wrong".

## What's next

- Let leaders write their own questions, or generate them from a lesson.
- Track misconceptions across sessions to show a chapter's progress over time.
- Support answers in more languages.

## Team

Soumya Vajahhala: solo builder (ML pipeline, evaluation, backend, front end, design).

## Links

- Code: https://github.com/V-SOUMYA/Unmuddle
- Live demo: add the Render URL after deploying
- Demo video: add the YouTube link
