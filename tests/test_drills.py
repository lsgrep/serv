import pytest

from servlab import drills


@pytest.mark.parametrize("factory", drills.ALL)
def test_every_drill_generates_and_reveals(factory, capsys):
    d = factory(seed=11)
    assert d.question
    assert d.steps or d.derivation is not None
    d.reveal()
    out = capsys.readouterr().out
    # Either shape of answer key must actually show the arithmetic.
    assert "ANSWER" in out or "WORKING" in out
    assert d.watch_for.split(".")[0][:20] in out


@pytest.mark.parametrize("factory", drills.ALL)
def test_no_drill_asks_you_to_recall_a_spec(factory):
    """The point of the rewrite: a drill hands you numbers, it does not name a
    card or a checkpoint and expect you to know its specifications."""
    q = factory(seed=3).question.lower()
    for name in ("h100", "a100", "t4 ", "llama", "qwen", "mistral", "gpt-2"):
        assert name not in q, f"{factory.__name__} leaks a spec you would have to recall: {name}"


@pytest.mark.parametrize("factory", (drills.kv_cache, drills.decode_ceiling,
                                     drills.ttft, drills.capacity))
def test_arithmetic_drills_show_a_full_derivation(factory, capsys):
    d = factory(seed=5)
    assert d.derivation is not None
    d.reveal()
    out = capsys.readouterr().out
    assert "GIVEN" in out and "WORKING" in out
    assert "<-" in out          # every given says where it was read from


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
