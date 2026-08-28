import pytest

from servlab.memory import GIB, logits_bytes, training_budget


def test_logits_are_gigabytes_per_sample_at_large_vocab():
    # 128K vocab x 4K sequence in fp32, plus the copy the loss makes.
    b = logits_bytes(1, 4096, 128_256) / GIB
    assert 3.5 < b < 4.5
    assert logits_bytes(1, 4096, 32_000) < logits_bytes(1, 4096, 128_256)


def test_a_fused_cross_entropy_removes_the_term():
    kw = dict(weight_bits=4, trainable_params=1.6e7, optimizer="adamw8bit", batch=1,
              seq_len=4096, n_layers=80, hidden=8192, vocab=128_256)
    with_logits = training_budget(70.6e9, **kw)
    fused = training_budget(70.6e9, fused_cross_entropy=True, **kw)
    assert with_logits.logits > 3 * GIB
    assert fused.logits == 0
    assert with_logits.total - fused.total == pytest.approx(with_logits.logits)


def test_qlora_removes_three_of_the_four_terms_but_not_activations():
    common = dict(batch=1, seq_len=2048, n_layers=80, hidden=8192)
    full = training_budget(70.6e9, weight_bits=16, optimizer="adamw", **common)
    qlora = training_budget(70.6e9, weight_bits=4, trainable_params=1.6e7,
                            optimizer="adamw8bit", **common)
    assert qlora.weights < full.weights / 3
    assert qlora.gradients < full.gradients / 100
    assert qlora.optimizer < full.optimizer / 100
    assert qlora.activations == full.activations      # the term QLoRA does not touch


def test_activations_are_why_qlora_ooms_on_sequence_length():
    kw = dict(weight_bits=4, trainable_params=1.6e7, optimizer="adamw8bit", batch=1,
              n_layers=80, hidden=8192, activation_checkpointing=False)
    short = training_budget(70.6e9, seq_len=1024, **kw)
    long = training_budget(70.6e9, seq_len=8192, **kw)
    assert long.activations == pytest.approx(8 * short.activations)
    assert long.total - short.total > 20 * GIB


def test_checkpointing_trades_time_for_most_of_the_activation_memory():
    kw = dict(weight_bits=4, trainable_params=1.6e7, optimizer="adamw8bit",
              batch=1, seq_len=4096, n_layers=80, hidden=8192)
    assert (training_budget(70.6e9, activation_checkpointing=True, **kw).activations
            < training_budget(70.6e9, activation_checkpointing=False, **kw).activations / 5)
