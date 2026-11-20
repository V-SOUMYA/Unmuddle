"""Step 3 of the pipeline: turn groups of wrong answers into misconceptions a teacher can act on.

For each question, one LLM call:
- merges groups that turn out to be the same misconception,
- names each misconception as the belief students hold,
- writes a 2-minute classroom fix,
- writes kind feedback for a learner who holds it,
- writes a follow-up question to check whether the misconception is gone.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from unmuddle.cluster import Cluster
from unmuddle.data import Answer, Question
from unmuddle.llm import LLM

EXPLAIN_PROMPT = """You help a student leader who teaches beginners. An algorithm has grouped the
WRONG answers to one question by meaning. Your job is to explain each group.

Question: {question}
Reference answer: {reference}
Misconceptions we already expect (use these names when they fit, may be empty): {known}

Groups of wrong answers (JSON):
{groups}

Instructions:
1. If two or more groups show the same misconception, merge them by listing all their
   group_ids in one item. Every group_id must appear in exactly one item.
2. For each misconception write, in simple English for beginners:
   - "title": the wrong belief, as students would say it, max 10 words.
     Example: "ChatGPT searches the internet for each answer".
   - "explanation": 1-2 sentences on why it is wrong.
   - "why_tempting": 1 sentence on why a beginner might believe it.
   - "fix": a 2-minute class activity, demo or analogy that fixes it. 2-4 sentences.
   - "learner_feedback": 2-3 kind sentences to a learner who holds this belief.
   - "follow_up_question": a NEW short question that checks if the belief is gone.
   - "follow_up_answer": the correct answer to the follow-up question, 1-2 sentences.
3. If a group mixes ideas, name it after its main idea.

Reply with JSON:
{{"misconceptions": [{{"group_ids": [0], "title": "", "explanation": "", "why_tempting": "",
"fix": "", "learner_feedback": "", "follow_up_question": "", "follow_up_answer": ""}}]}}
"""

FOLLOW_UP_PROMPT = """A learner believed: "{title}".
To check whether this belief is gone, they were asked: {question}
Correct answer: {reference}
Learner's answer: {answer}

Is the learner's answer correct, and is the wrong belief gone? Judge the idea, not the wording.
Reply with JSON: {{"fixed": true or false, "feedback": "1-2 kind sentences to the learner"}}
"""

TEXT_FIELDS = [
    "title",
    "explanation",
    "why_tempting",
    "fix",
    "learner_feedback",
    "follow_up_question",
    "follow_up_answer",
]


@dataclass
class Misconception:
    question_id: str
    misconception_id: str
    title: str
    explanation: str = ""
    why_tempting: str = ""
    fix: str = ""
    learner_feedback: str = ""
    follow_up_question: str = ""
    follow_up_answer: str = ""
    student_ids: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.student_ids)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["size"] = self.size
        return d


def _groups_payload(clusters: list[Cluster], max_answers: int = 12) -> str:
    groups = [
        {
            "group_id": c.cluster_id,
            "size": c.size,
            "answers": [a.text for a in c.answers[:max_answers]],
        }
        for c in clusters
    ]
    return json.dumps(groups, ensure_ascii=False, indent=1)


def _parse(reply, clusters: list[Cluster]) -> list[tuple[list[Cluster], dict]]:
    """Match LLM items back to clusters. Never lose a cluster, never use one twice."""
    by_id = {c.cluster_id: c for c in clusters}
    used: set[int] = set()
    items = reply.get("misconceptions", []) if isinstance(reply, dict) else []
    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        ids = []
        for raw in item.get("group_ids", []):
            try:
                gid = int(raw)
            except (TypeError, ValueError):
                continue
            if gid in by_id and gid not in used:
                ids.append(gid)
                used.add(gid)
        if ids:
            out.append(([by_id[i] for i in ids], item))
    for c in clusters:  # anything the LLM skipped still reaches the teacher
        if c.cluster_id not in used:
            out.append(([c], {"title": f"Other wrong idea: \"{c.examples[0].text}\""}))
    return out


def explain_question(question: Question, clusters: list[Cluster], llm: LLM) -> list[Misconception]:
    """Name and explain the misconceptions behind one question's clusters. Biggest first."""
    if not clusters:
        return []
    prompt = EXPLAIN_PROMPT.format(
        question=question.text,
        reference=question.reference_answer,
        known=json.dumps(question.known_misconceptions, ensure_ascii=False),
        groups=_groups_payload(clusters),
    )
    reply = llm.json(prompt)

    misconceptions = []
    for group, item in _parse(reply, clusters):
        answers: list[Answer] = [a for c in group for a in c.answers]
        examples = [e.text for c in group for e in c.examples][:3]
        fields = {k: str(item.get(k, "")).strip() for k in TEXT_FIELDS}
        fields["title"] = fields["title"] or "Unnamed misconception"
        misconceptions.append(
            Misconception(
                question_id=question.question_id,
                misconception_id="",
                student_ids=[a.student_id for a in answers],
                examples=examples,
                **fields,
            )
        )
    misconceptions.sort(key=lambda m: -m.size)
    for i, m in enumerate(misconceptions):
        m.misconception_id = f"{question.question_id}-m{i}"
    return misconceptions


def check_follow_up(misconception: Misconception, answer: str, llm: LLM) -> dict:
    """Grade a learner's answer to the follow-up question. Returns {"fixed", "feedback"}."""
    prompt = FOLLOW_UP_PROMPT.format(
        title=misconception.title,
        question=misconception.follow_up_question,
        reference=misconception.follow_up_answer,
        answer=answer,
    )
    reply = llm.json(prompt)
    if not isinstance(reply, dict):
        reply = {}
    return {"fixed": bool(reply.get("fixed", False)), "feedback": str(reply.get("feedback", ""))}
