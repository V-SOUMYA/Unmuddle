"""A fake LLM that answers every prompt type the pipeline sends, for offline tests."""

import json
import re


def _json_after(prompt: str, marker: str):
    block = prompt.split(marker, 1)[1]
    start = block.index("[")
    depth = 0
    for i, ch in enumerate(block[start:], start):
        depth += ch == "["
        depth -= ch == "]"
        if depth == 0:
            return json.loads(block[start : i + 1])
    raise ValueError("no JSON list found")


def fake_respond(prompt: str):
    if prompt.startswith("You are grading short student answers."):
        # Uses the reference answer's first word list as "keywords" for correctness.
        answers = _json_after(prompt, "Student answers (JSON list")
        keys = ("predict", "memoriz", "overfit", "biased", "proxy", "is null", "having", "3 rows")
        return [{"id": a["id"], "correct": any(k in a["answer"].lower() for k in keys)} for a in answers]

    if prompt.startswith("You help a student leader"):
        groups = _json_after(prompt, "Groups of wrong answers (JSON):")
        ids = [g["group_id"] for g in groups]
        # Merge the first two groups to exercise merging, name the rest.
        items = []
        if len(ids) >= 2:
            items.append({"group_ids": ids[:2], "title": "Merged myth", "fix": "Do a demo."})
            ids = ids[2:]
        items += [{"group_ids": [i], "title": f"Myth {i}", "fix": "Draw it."} for i in ids]
        for item in items:
            item.update(
                explanation="Because.",
                why_tempting="It sounds right.",
                learner_feedback="Good try.",
                follow_up_question="New question?",
                follow_up_answer="New answer.",
            )
        return {"misconceptions": items}

    if prompt.startswith("A learner believed"):
        answer = re.search(r"Learner's answer: (.*)", prompt).group(1)
        return {"fixed": "predict" in answer.lower(), "feedback": "Thanks!"}

    raise ValueError("unexpected prompt")
