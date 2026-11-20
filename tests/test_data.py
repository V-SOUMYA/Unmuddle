from unmuddle.data import group_by_question, load_answers, load_sample


def test_ai_sample_loads():
    questions, answers = load_sample("ai")
    assert set(questions) == {"ai1", "ai2", "ai3"}
    assert len(answers) == 36
    assert all(a.question_id in questions for a in answers)
    assert all(a.is_correct is not None for a in answers)


def test_sql_sample_loads():
    questions, answers = load_sample("sql")
    assert set(questions) == {"q1", "q2", "q3"}
    assert len(answers) == 36


def test_group_by_question_keeps_order():
    _, answers = load_sample("ai")
    groups = group_by_question(answers)
    assert [len(g) for g in groups.values()] == [12, 12, 12]
    assert groups["ai1"][0].student_id == "l01"


def test_missing_label_is_none(tmp_path):
    csv = tmp_path / "answers.csv"
    csv.write_text("student_id,question_id,answer\nx1,q1,hello\nx2,q1,\n")
    answers = load_answers(csv)
    assert answers[0].is_correct is None
    assert answers[1].text == ""
