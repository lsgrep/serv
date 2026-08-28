import pytest

from servlab import drills


@pytest.mark.parametrize("factory", drills.ALL)
def test_every_drill_generates_and_reveals(factory, capsys):
    d = factory(seed=11)
    assert d.question and d.steps
    d.reveal()
    out = capsys.readouterr().out
    assert "ANSWER" in out
    assert d.answer.split("·")[0].strip()[:6] in out


def test_drills_are_seeded_so_a_session_is_reproducible():
    assert drills.kv_cache(seed=5).question == drills.kv_cache(seed=5).question
    assert drills.kv_cache(seed=5).question != drills.kv_cache(seed=6).question


def test_kv_drill_arithmetic_is_self_consistent(capsys):
    d = drills.kv_cache(seed=42)
    capsys.readouterr()
    # The worked steps must contain the multiplication, not just the result —
    # the method is the thing being rehearsed.
    assert any("2 (K and V)" in s for s in d.steps)
    assert "KV heads, not attention heads" in d.watch_for


def test_drill_set_returns_distinct_drills():
    ds = drills.drill_set(n=4, seed=3)
    assert len(ds) == 4
    assert len({d.title for d in ds}) == 4


def test_flashcards_hide_the_answer_until_asked(capsys):
    drills.flashcards(n=5, seed=1)
    hidden = capsys.readouterr().out
    assert "rerun with reveal=True" in hidden
    drills.flashcards(n=5, seed=1, reveal=True)
    assert "->" in capsys.readouterr().out
