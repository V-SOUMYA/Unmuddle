from unmuddle.data import load_lessons


def test_twelve_lessons_in_order():
    lessons = load_lessons()
    assert [l.number for l in lessons] == list(range(1, 13))


def test_every_question_is_complete_and_unique():
    ids = []
    for lesson in load_lessons():
        assert lesson.questions, lesson.title
        for q in lesson.questions:
            assert q.text and q.reference_answer and q.topic
            assert len(q.known_misconceptions) >= 2
            assert q.question_id.startswith(lesson.lesson_id)
            ids.append(q.question_id)
    assert len(ids) == len(set(ids))
