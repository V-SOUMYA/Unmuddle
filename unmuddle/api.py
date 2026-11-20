"""Web API for live sessions.

Flow:
1. A leader picks a lesson -> POST /api/sessions -> gets a 6-letter code and a QR code.
2. Learners join on their phones and submit answers.
3. The leader presses Analyze -> the pipeline groups wrong answers by misconception.
4. Each learner sees personal feedback and a follow-up question; their result feeds back
   into the dashboard as "fixed" counts.

Run locally:
    uvicorn unmuddle.api:app --reload
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import segno
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from unmuddle.config import ROOT
from unmuddle.data import Lesson, load_lessons, load_sample
from unmuddle.embed import Embedder
from unmuddle.explain import Misconception, check_follow_up
from unmuddle.llm import LLM
from unmuddle.pipeline import analyze
from unmuddle.store import Store

WEB_DIR = ROOT / "web"
MAX_ANSWER_CHARS = 1000
MAX_NAME_CHARS = 40


class NewSession(BaseModel):
    lesson_id: str


class Submission(BaseModel):
    name: str = Field(min_length=1, max_length=MAX_NAME_CHARS)
    answers: dict[str, str]


class FollowUpAnswer(BaseModel):
    misconception_id: str
    answer: str = Field(min_length=1, max_length=MAX_ANSWER_CHARS)


def _default_llm() -> LLM:
    from unmuddle.llm import GeminiLLM

    return GeminiLLM()


def _default_embedder() -> Embedder:
    from unmuddle.embed import get_embedder

    return get_embedder(os.getenv("UNMUDDLE_EMBEDDER", "minilm"))


def _public_question(q) -> dict:
    return {"question_id": q.question_id, "topic": q.topic, "text": q.text}


def _lesson_summary(lesson: Lesson) -> dict:
    return {
        "lesson_id": lesson.lesson_id,
        "number": lesson.number,
        "title": lesson.title,
        "questions": [_public_question(q) for q in lesson.questions],
    }


def create_app(
    db_path: str | Path | None = None,
    llm_factory: Callable[[], LLM] = _default_llm,
    embedder_factory: Callable[[], Embedder] = _default_embedder,
) -> FastAPI:
    app = FastAPI(title="Unmuddle", description="Find out why students get it wrong.")
    store = Store(db_path or os.getenv("UNMUDDLE_DB", ROOT / "unmuddle.db"))
    lessons = {l.lesson_id: l for l in load_lessons()}
    cache: dict[str, object] = {}

    def llm() -> LLM:
        if "llm" not in cache:
            cache["llm"] = llm_factory()
        return cache["llm"]

    def embedder() -> Embedder:
        if "embedder" not in cache:
            cache["embedder"] = embedder_factory()
        return cache["embedder"]

    def session_or_404(code: str) -> dict:
        session = store.get_session(code)
        if session is None:
            raise HTTPException(404, "No session with that code.")
        return session

    def find_misconception(report: dict, misconception_id: str) -> Misconception | None:
        for q in report["questions"]:
            for m in q["misconceptions"]:
                if m["misconception_id"] == misconception_id:
                    fields = {k: v for k, v in m.items() if k != "size"}
                    return Misconception(**fields)
        return None

    # Lessons and sessions -------------------------------------------------

    @app.get("/api/health")
    def health():
        return {"ok": True}

    @app.get("/api/lessons")
    def list_lessons():
        return [_lesson_summary(l) for l in lessons.values()]

    @app.post("/api/sessions")
    def new_session(body: NewSession):
        lesson = lessons.get(body.lesson_id)
        if lesson is None:
            raise HTTPException(404, "Unknown lesson.")
        code = store.create_session(f"Lesson {lesson.number}: {lesson.title}", lesson.questions)
        return {"code": code}

    @app.post("/api/demo")
    def demo_session():
        """A ready-made session with 12 learners' answers, so anyone can try the dashboard."""
        questions, answers = load_sample("ai")
        code = store.create_session("Demo: How AI works", list(questions.values()))
        by_learner: dict[str, dict[str, str]] = {}
        for a in answers:
            by_learner.setdefault(a.student_id, {})[a.question_id] = a.text
        for i, (_, learner_answers) in enumerate(sorted(by_learner.items()), 1):
            store.add_responses(code, f"Learner {i}", learner_answers)
        return {"code": code}

    @app.get("/api/sessions/{code}")
    def get_session(code: str):
        s = session_or_404(code)
        return {
            "code": s["code"],
            "title": s["title"],
            "questions": [_public_question(q) for q in s["questions"]],
            "analyzed": s["report"] is not None,
            **store.stats(s["code"]),
        }

    @app.get("/api/sessions/{code}/qr.svg")
    def qr(code: str, request: Request):
        s = session_or_404(code)
        url = f"{str(request.base_url).rstrip('/')}/join.html?code={s['code']}"
        svg = segno.make(url, error="m").svg_inline(scale=8, border=2)
        return Response(svg, media_type="image/svg+xml")

    # Learner side ---------------------------------------------------------

    @app.post("/api/sessions/{code}/responses")
    def submit(code: str, body: Submission):
        s = session_or_404(code)
        valid = {q.question_id for q in s["questions"]}
        answers = {
            qid: text[:MAX_ANSWER_CHARS] for qid, text in body.answers.items() if qid in valid and text.strip()
        }
        if not answers:
            raise HTTPException(400, "Please answer at least one question.")
        learner_id = store.add_responses(s["code"], body.name.strip(), answers)
        return {"learner_id": learner_id}

    @app.get("/api/sessions/{code}/learners/{learner_id}/feedback")
    def learner_feedback(code: str, learner_id: str):
        s = session_or_404(code)
        mine = store.learner_answers(s["code"], learner_id)
        if not mine:
            raise HTTPException(404, "No answers from this learner.")
        report = s["report"]
        followups = store.learner_followups(s["code"], learner_id)
        items = []
        for q in s["questions"]:
            if q.question_id not in mine:
                continue
            item = {"question_id": q.question_id, "topic": q.topic, "question": q.text, "your_answer": mine[q.question_id]}
            if report is not None:
                m = None
                for qr in report["questions"]:
                    for cand in qr["misconceptions"] if qr["question_id"] == q.question_id else []:
                        if learner_id in cand["student_ids"]:
                            m = cand
                analyzed_here = any(qr["question_id"] == q.question_id for qr in report["questions"])
                item["correct"] = analyzed_here and m is None
                item["reference_answer"] = q.reference_answer if analyzed_here else None
                if m is not None:
                    item["misconception"] = {
                        k: m[k]
                        for k in ("misconception_id", "title", "explanation", "learner_feedback", "follow_up_question")
                    }
                    item["follow_up"] = followups.get(m["misconception_id"])
            items.append(item)
        return {"ready": report is not None, "items": items}

    @app.post("/api/sessions/{code}/learners/{learner_id}/followup")
    async def answer_followup(code: str, learner_id: str, body: FollowUpAnswer):
        s = session_or_404(code)
        if s["report"] is None:
            raise HTTPException(409, "This session has not been analyzed yet.")
        m = find_misconception(s["report"], body.misconception_id)
        if m is None or learner_id not in m.student_ids:
            raise HTTPException(404, "No such follow-up for this learner.")
        result = await run_in_threadpool(check_follow_up, m, body.answer, llm())
        store.add_followup(s["code"], learner_id, m.misconception_id, body.answer, result["fixed"], result["feedback"])
        return result

    # Leader side ----------------------------------------------------------

    @app.post("/api/sessions/{code}/analyze")
    async def run_analysis(code: str):
        s = session_or_404(code)
        answers = store.answers(s["code"])
        if not answers:
            raise HTTPException(400, "No answers yet.")
        questions = {q.question_id: q for q in s["questions"]}
        report = await run_in_threadpool(analyze, questions, answers, llm(), embedder())
        store.save_report(s["code"], report.to_dict())
        return get_report(code)

    @app.get("/api/sessions/{code}/report")
    def get_report(code: str):
        s = session_or_404(code)
        if s["report"] is None:
            raise HTTPException(404, "Not analyzed yet.")
        stats = store.followup_stats(s["code"])
        report = s["report"]
        for q in report["questions"]:
            for m in q["misconceptions"]:
                m["follow_up_stats"] = stats.get(m["misconception_id"], {"attempted": 0, "fixed": 0})
        live = store.stats(s["code"])
        return {
            "code": s["code"],
            "title": s["title"],
            "analyzed_at": s["analyzed_at"],
            "new_answers_since_analysis": bool(live["last_answer_at"] and live["last_answer_at"] > s["analyzed_at"]),
            **live,
            **report,
        }

    # Web pages ------------------------------------------------------------

    if WEB_DIR.exists():
        app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    else:

        @app.get("/")
        def root():
            return RedirectResponse("/docs")

    return app


app = create_app()
