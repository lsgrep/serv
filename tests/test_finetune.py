import json

import pytest

from servlab import finetune as ft
from servlab import rag


@pytest.fixture
def chunks():
    return rag.policy_corpus()[0]


def test_training_and_heldout_questions_never_overlap():
    # The whole experiment depends on this. If a held-out question appears in
    # training, the knowledge result is memorisation wearing a disguise.
    train = {e.prompt for e in ft.knowledge_examples("train")}
    held = {e.prompt for e in ft.knowledge_examples("test")}
    assert train and held
    assert not (train & held)


def test_held_out_questions_ask_about_the_same_facts():
    # Different words, same facts — otherwise the comparison measures topic
    # difficulty rather than generalisation.
    train_facts = {e.meta["fact"] for e in ft.knowledge_examples("train")}
    held_facts = {e.meta["fact"] for e in ft.knowledge_examples("test")}
    assert held_facts <= train_facts
    assert len(held_facts) >= 10


def test_held_out_phrasings_share_few_words_with_training(chunks):
    # A paraphrase that reuses the training wording is not a paraphrase.
    for fact in ft.FACTS:
        train_words = set()
        for q in fact.train_questions:
            train_words |= set(q.lower().split())
        for q in fact.heldout_questions:
            overlap = set(q.lower().split()) & train_words
            content = {w for w in overlap if len(w) > 4}
            assert len(content) <= 2, (fact.id, q, content)


def test_format_examples_carry_a_real_passage_and_valid_json(chunks):
    examples = ft.format_examples(chunks, "train")
    assert examples
    ids = {c.id for c in chunks}
    for e in examples:
        obj = json.loads(e.completion)
        assert set(ft.REQUIRED_KEYS) <= set(obj)
        assert obj["citation"] in ids
        assert obj["confidence"] in ft.CONFIDENCE_VALUES
        assert "Context [" in e.prompt


def test_knowledge_examples_are_closed_book():
    # No passage in the prompt: that is what makes it a knowledge test rather
    # than a reading test.
    for e in ft.knowledge_examples("train"):
        assert "Context [" not in e.prompt
        assert e.system == ft.KNOWLEDGE_SYSTEM


def test_strict_json_is_scored_separately_from_json_somewhere_in_the_text():
    clean = '{"answer": "30 days", "citation": "refunds#0", "confidence": "high"}'
    chatty = 'Sure! Here you go:\n{"answer": "30 days", "citation": "refunds#0", "confidence": "high"}'

    assert ft.score_format(clean)["strict"]
    assert not ft.score_format(chatty)["strict"]
    assert ft.score_format(chatty)["parses"]
    assert ft.score_format(chatty)["extra_prose"]
    # A downstream parser only cares about the strict column.
    assert ft.format_rate([clean, chatty])["usable"] == pytest.approx(0.5)
    assert ft.format_rate([clean, chatty])["parses_anywhere"] == pytest.approx(1.0)


def test_format_scorer_catches_missing_keys_and_bad_confidence():
    assert not ft.score_format('{"answer": "30 days"}')["keys_ok"]
    bad = '{"answer": "x", "citation": "refunds#0", "confidence": "very sure"}'
    assert ft.score_format(bad)["keys_ok"]
    assert not ft.score_format(bad)["confidence_ok"]


def test_format_scorer_flags_invented_citations(chunks):
    ids = [c.id for c in chunks]
    good = '{"answer": "x", "citation": "%s", "confidence": "high"}' % ids[0]
    bad = '{"answer": "x", "citation": "handbook#9", "confidence": "high"}'
    assert ft.score_format(good, ids)["citation_valid"]
    assert not ft.score_format(bad, ids)["citation_valid"]


def test_format_scorer_survives_garbage():
    for junk in ("", None, "not json at all", "{unclosed", "[1, 2, 3]"):
        s = ft.score_format(junk)
        assert s["strict"] is False or junk == "[1, 2, 3]"


def test_knowledge_scoring_is_lenient_on_wording_strict_on_the_fact():
    e = next(x for x in ft.knowledge_examples("test") if x.meta["fact"] == "refund-window")
    assert ft.score_knowledge("You have 30 days from purchase.", e)
    assert ft.score_knowledge("thirty days", e)              # accepted surface form
    assert not ft.score_knowledge("You have 90 days.", e)
    assert not ft.score_knowledge("I don't know.", e)


def test_evaluate_knowledge_runs_any_generator():
    examples = ft.knowledge_examples("test")[:4]
    perfect = ft.evaluate_knowledge(examples, lambda p, s: examples[0].completion
                                    if p == examples[0].prompt else "no idea")
    assert perfect["n"] == 4
    assert 0 < perfect["accuracy"] < 1


def test_capability_probe_detects_a_model_that_only_emits_json():
    # The failure mode of narrow fine-tuning: the format eats the model.
    broken = ft.capability_check(lambda p, s: '{"answer": "x", "citation": "y", "confidence": "high"}')
    assert broken["accuracy"] == 0.0
    healthy = ft.capability_check(lambda p, s: dict(ft.CAPABILITY_PROBES).get(p, ""))
    assert healthy["accuracy"] == 1.0


def test_augmentation_repeats_without_mutating_the_source():
    base = ft.knowledge_examples("train")
    aug = ft.augment(base, factor=4)
    assert len(aug) == 4 * len(base)
    assert {e.prompt for e in aug} == {e.prompt for e in base}
    assert base[0].meta is not aug[0].meta


def test_to_dataset_uses_the_tokenizer_template_when_given():
    class FakeTok:
        chat_template = "x"

        def apply_chat_template(self, messages, tokenize=False):
            return "TEMPLATED:" + messages[-1]["content"]

    rows = ft.to_dataset(ft.knowledge_examples("train")[:2], tokenizer=FakeTok())
    assert all(r["text"].startswith("TEMPLATED:") for r in rows)
    plain = ft.to_dataset(ft.knowledge_examples("train")[:2])
    assert all("<|im_start|>" in r["text"] for r in plain)


def test_verdict_names_the_conclusion_when_the_experiment_shows_it():
    v = ft.verdict(fine_tuned_heldout=0.25, base_heldout=0.20, rag_heldout=0.85,
                   format_gain=0.7)
    assert "did not install knowledge" in v

    v2 = ft.verdict(fine_tuned_heldout=0.70, base_heldout=0.20, rag_heldout=0.85,
                    format_gain=0.7)
    assert "genuinely improved" in v2
    assert "Report both numbers" in v2


def test_comparison_table_renders_missing_columns_as_dashes():
    out = ft.comparison_table([{"name": "base", "format_usable": 0.1, "held_out": 0.2},
                               {"name": "fine-tuned", "format_usable": 0.9}])
    assert "base" in out and "fine-tuned" in out
    assert "-" in out
