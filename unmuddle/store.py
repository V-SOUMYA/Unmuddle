"""SQLite storage for live sessions, learner answers, reports and follow-up checks.

Plain sqlite3, one file, no server: easy to run anywhere.
"""

from __future__ import annotations

import json
import secrets
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from unmuddle.data import Answer, Question

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    code TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    questions_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    report_json TEXT,
    analyzed_at REAL
);
CREATE TABLE IF NOT EXISTS responses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT NOT NULL,
    learner_id TEXT NOT NULL,
    name TEXT NOT NULL,
    question_id TEXT NOT NULL,
    answer TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS followups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT NOT NULL,
    learner_id TEXT NOT NULL,
    misconception_id TEXT NOT NULL,
    answer TEXT NOT NULL,
    fixed INTEGER NOT NULL,
    feedback TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_responses_code ON responses(code);
CREATE INDEX IF NOT EXISTS idx_followups_code ON followups(code);
"""

# No 0/O or 1/I, so codes are easy to read out loud.
CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


def _question_dict(q: Question) -> dict:
    return {
        "question_id": q.question_id,
        "topic": q.topic,
        "text": q.text,
        "reference_answer": q.reference_answer,
        "known_misconceptions": q.known_misconceptions,
    }


class Store:
    def __init__(self, path: str | Path = ":memory:"):
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock:
            self.conn.executescript(SCHEMA)

    def _new_code(self) -> str:
        while True:
            code = "".join(secrets.choice(CODE_ALPHABET) for _ in range(6))
            if self.get_session(code) is None:
                return code

    # Sessions -------------------------------------------------------------

    def create_session(self, title: str, questions: list[Question]) -> str:
        code = self._new_code()
        payload = json.dumps([_question_dict(q) for q in questions])
        with self.lock, self.conn:
            self.conn.execute(
                "INSERT INTO sessions (code, title, questions_json, created_at) VALUES (?, ?, ?, ?)",
                (code, title, payload, time.time()),
            )
        return code

    def get_session(self, code: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM sessions WHERE code = ?", (code.upper(),)).fetchone()
        if row is None:
            return None
        return {
            "code": row["code"],
            "title": row["title"],
            "questions": [Question(**q) for q in json.loads(row["questions_json"])],
            "created_at": row["created_at"],
            "report": json.loads(row["report_json"]) if row["report_json"] else None,
            "analyzed_at": row["analyzed_at"],
        }

    def save_report(self, code: str, report: dict) -> None:
        with self.lock, self.conn:
            self.conn.execute(
                "UPDATE sessions SET report_json = ?, analyzed_at = ? WHERE code = ?",
                (json.dumps(report), time.time(), code),
            )

    # Responses ------------------------------------------------------------

    def add_responses(self, code: str, name: str, answers: dict[str, str], learner_id: str | None = None) -> str:
        learner_id = learner_id or uuid.uuid4().hex[:12]
        now = time.time()
        rows = [(code, learner_id, name, qid, text.strip(), now) for qid, text in answers.items()]
        with self.lock, self.conn:
            self.conn.executemany(
                "INSERT INTO responses (code, learner_id, name, question_id, answer, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )
        return learner_id

    def answers(self, code: str) -> list[Answer]:
        rows = self.conn.execute(
            "SELECT learner_id, question_id, answer FROM responses WHERE code = ? ORDER BY id", (code,)
        ).fetchall()
        return [Answer(r["learner_id"], r["question_id"], r["answer"]) for r in rows]

    def learner_answers(self, code: str, learner_id: str) -> dict[str, str]:
        rows = self.conn.execute(
            "SELECT question_id, answer FROM responses WHERE code = ? AND learner_id = ?", (code, learner_id)
        ).fetchall()
        return {r["question_id"]: r["answer"] for r in rows}

    def stats(self, code: str) -> dict:
        row = self.conn.execute(
            "SELECT COUNT(DISTINCT learner_id) AS learners, COUNT(*) AS answers, MAX(created_at) AS last "
            "FROM responses WHERE code = ?",
            (code,),
        ).fetchone()
        return {"n_learners": row["learners"], "n_answers": row["answers"], "last_answer_at": row["last"]}

    # Follow-ups -----------------------------------------------------------

    def add_followup(self, code: str, learner_id: str, misconception_id: str, answer: str, fixed: bool, feedback: str) -> None:
        with self.lock, self.conn:
            self.conn.execute(
                "INSERT INTO followups (code, learner_id, misconception_id, answer, fixed, feedback, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (code, learner_id, misconception_id, answer, int(fixed), feedback, time.time()),
            )

    def learner_followups(self, code: str, learner_id: str) -> dict[str, dict]:
        """Latest follow-up result per misconception for one learner."""
        rows = self.conn.execute(
            "SELECT misconception_id, fixed, feedback FROM followups "
            "WHERE code = ? AND learner_id = ? ORDER BY id",
            (code, learner_id),
        ).fetchall()
        return {r["misconception_id"]: {"fixed": bool(r["fixed"]), "feedback": r["feedback"]} for r in rows}

    def followup_stats(self, code: str) -> dict[str, dict]:
        """Per misconception: how many learners tried the follow-up, and how many fixed it.

        Uses each learner's latest attempt.
        """
        rows = self.conn.execute(
            "SELECT misconception_id, learner_id, fixed FROM followups WHERE code = ? ORDER BY id", (code,)
        ).fetchall()
        latest: dict[tuple[str, str], bool] = {}
        for r in rows:
            latest[(r["misconception_id"], r["learner_id"])] = bool(r["fixed"])
        stats: dict[str, dict] = {}
        for (mid, _), fixed in latest.items():
            s = stats.setdefault(mid, {"attempted": 0, "fixed": 0})
            s["attempted"] += 1
            s["fixed"] += int(fixed)
        return stats
