import pytest

from servlab import rag


def test_chunks_overlap_so_facts_do_not_fall_off_boundaries():
    text = " ".join(f"w{i}" for i in range(100))
    chunks = rag.chunk_text(text, size=30, overlap=10)
    assert len(chunks) > 3
    first, second = chunks[0].text.split(), chunks[1].text.split()
    assert first[-10:] == second[:10]      # the overlap is real, not nominal


def test_chunking_rejects_impossible_settings():
    with pytest.raises(ValueError):
        rag.chunk_text("a b c", size=10, overlap=10)


def test_bm25_finds_the_keyword_document():
    chunks, _ = rag.policy_corpus()
    hits = rag.BM25(chunks).search("audit logs retention", k=3)
    assert hits and hits[0][0].doc_id == "security"


def test_retrieval_metrics_are_measured_separately_from_answers():
    # recall@k caps everything downstream: a passage that was never retrieved
    # cannot be synthesised from, no matter which model reads the context.
    assert rag.recall_at_k(["a", "b", "c"], ["c"], k=3) == 1.0
    assert rag.recall_at_k(["a", "b", "c"], ["c"], k=2) == 0.0
    assert rag.mrr(["a", "b", "c"], ["b"]) == 0.5
    assert rag.ndcg_at_k(["a"], ["a"], k=3) == pytest.approx(1.0)


def test_the_policy_corpus_is_mostly_retrievable_with_lexical_search_alone():
    chunks, cases = rag.policy_corpus()
    retriever = rag.HybridRetriever(chunks)
    answerable = [c for c in cases if c["answerable"]]
    result = rag.evaluate_retrieval(retriever, answerable, k=3)
    # Good but not perfect is the point: the misses are the paraphrase queries,
    # which is exactly what a reranker or a dense retriever is for.
    assert 0.6 < result["hit_rate"] < 1.0


def test_reranking_reorders_by_query_coverage():
    chunks, _ = rag.policy_corpus()
    wide = rag.BM25(chunks).search("service credit uptime below 95 percent", k=8)
    reranked = rag.rerank_by_overlap("service credit uptime below 95 percent", wide, top_k=3)
    assert len(reranked) == 3
    assert reranked[0][0].doc_id == "sla"


def test_fusion_keeps_what_either_retriever_ranked_well():
    chunks, _ = rag.policy_corpus()
    a = [(chunks[0], 1.0), (chunks[1], 0.5)]
    b = [(chunks[1], 9.9), (chunks[2], 1.1)]
    fused = rag.reciprocal_rank_fusion([a, b], top_k=3)
    ids = [c.id for c, _ in fused]
    # chunk[1] is ranked by both, so it wins despite topping neither list
    assert ids[0] == chunks[1].id


def test_triage_separates_retrieval_from_synthesis_failures():
    cases = (
        [{"gold_chunk_ids": ["g"], "retrieved_ids": ["x", "y"], "answer_correct": False}] * 34
        + [{"gold_chunk_ids": ["g"], "retrieved_ids": ["g"], "answer_correct": False}] * 6
    )
    result = rag.triage(cases)
    assert result["retrieval_misses"] == 34
    assert result["synthesis_misses"] == 6
    assert result["retrieval_share_of_failures"] == pytest.approx(34 / 40)

    rec = rag.triage_recommendation(result)
    assert "retrieval problem" in rec
    assert "fine-tuning would not have moved" in rec.lower()


def test_triage_says_the_opposite_when_retrieval_is_fine():
    cases = (
        [{"gold_chunk_ids": ["g"], "retrieved_ids": ["g"], "answer_correct": False}] * 18
        + [{"gold_chunk_ids": ["g"], "retrieved_ids": ["x"], "answer_correct": False}] * 2
    )
    rec = rag.triage_recommendation(rag.triage(cases))
    assert "synthesis problem" in rec
    assert "last resort" in rec


def test_triage_refuses_to_conclude_from_no_failures():
    result = rag.triage([{"gold_chunk_ids": ["g"], "retrieved_ids": ["g"], "answer_correct": True}])
    assert "Widen the eval set" in rag.triage_recommendation(result)


def test_corpus_includes_unanswerable_questions():
    # The failure mode nobody tests: confident answers to questions the corpus
    # cannot support.
    _, cases = rag.policy_corpus()
    assert any(not c["answerable"] for c in cases)
