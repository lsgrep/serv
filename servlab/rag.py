"""Retrieval: chunking, hybrid search, and the failure triage that matters.

The exec conversation this module exists to support:

    "The answers are wrong, so the model is bad, so let's fine-tune."

Fine-tuning teaches style and format. It does not install knowledge the system
was never handed — `servlab.finetune` and lab 10 measure that rather than
assuming it. Before touching a model you pull the failures and split them
into two buckets — **the right passage was never retrieved** (a retrieval
problem, cheap to fix) versus **it was retrieved and the answer is still wrong**
(a synthesis problem, and the only bucket where a model change is even the right
category of fix).

`triage()` does that split. Being able to say "34 of these 40 are retrieval
misses" is worth more than any argument, because it relocates the problem
without contradicting anyone's experience of it.

Pure Python, no dependencies: BM25 and the metrics run anywhere, and a dense
retriever is used only if sentence-transformers happens to be installed.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text) -> list:
    return _WORD.findall((text or "").lower())


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------


@dataclass
class Chunk:
    id: str
    text: str
    doc_id: str = ""
    index: int = 0
    meta: dict = field(default_factory=dict)


def chunk_text(text, doc_id="doc", size=200, overlap=40, unit="words") -> list:
    """Split into overlapping chunks.

    The two knobs and what they trade:

    * **size** — too small and a passage loses the context that makes it
      answerable; too large and the embedding averages several topics into a
      vector that matches none of them well. 150-300 words is a reasonable
      first guess for prose, and it is a guess: measure recall@k, do not debate.
    * **overlap** — insurance against a fact landing on a boundary. Roughly 10-25%
      of size. Zero overlap is the most common cause of a fact that is
      *in the corpus* and never retrievable.
    """
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("need size > 0 and 0 <= overlap < size")
    units = text.split() if unit == "words" else list(text)
    join = " " if unit == "words" else ""
    chunks, start, i = [], 0, 0
    step = size - overlap
    while start < len(units):
        piece = join.join(units[start:start + size])
        if piece.strip():
            chunks.append(Chunk(id=f"{doc_id}#{i}", text=piece, doc_id=doc_id, index=i))
            i += 1
        start += step
    return chunks


def chunk_documents(docs, **kw) -> list:
    """`docs` is {doc_id: text}."""
    out = []
    for doc_id, text in docs.items():
        out.extend(chunk_text(text, doc_id=doc_id, **kw))
    return out


# --------------------------------------------------------------------------
# Lexical retrieval (BM25)
# --------------------------------------------------------------------------


class BM25:
    """Okapi BM25 in forty lines.

    Worth having in front of you rather than behind an import: it is a term
    frequency saturating (`k1`) and a length normalisation (`b`), and it beats
    a mediocre embedding model on keyword-ish queries — product codes, error
    strings, policy numbers — which is exactly the traffic enterprise RAG gets.
    """

    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks = list(chunks)
        self.k1, self.b = k1, b
        self.docs = [tokenize(c.text) for c in self.chunks]
        self.lengths = [len(d) for d in self.docs]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        self.tf = [Counter(d) for d in self.docs]
        df = Counter()
        for d in self.docs:
            df.update(set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def score(self, query) -> list:
        q = tokenize(query)
        scores = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / (self.avg_len or 1))
            for term in q:
                f = tf.get(term)
                if not f:
                    continue
                s += self.idf.get(term, 0.0) * f * (self.k1 + 1) / (f + norm)
            scores.append(s)
        return scores

    def search(self, query, k=5) -> list:
        scores = self.score(query)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(self.chunks[i], scores[i]) for i in order if scores[i] > 0]


# --------------------------------------------------------------------------
# Dense retrieval (optional) and fusion
# --------------------------------------------------------------------------


class DenseIndex:
    """Embedding search, if sentence-transformers is available.

    Kept optional on purpose: every lesson in this module lands with BM25 alone,
    and a lab that cannot run offline is a lab you skip.
    """

    def __init__(self, chunks, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.chunks = list(chunks)
        self.model = SentenceTransformer(model_name)
        self.vectors = self.model.encode([c.text for c in self.chunks],
                                         normalize_embeddings=True, show_progress_bar=False)

    def search(self, query, k=5) -> list:
        import numpy as np

        q = self.model.encode([query], normalize_embeddings=True)[0]
        sims = np.asarray(self.vectors) @ q
        order = np.argsort(-sims)[:k]
        return [(self.chunks[i], float(sims[i])) for i in order]


def reciprocal_rank_fusion(result_lists, k=60, top_k=5) -> list:
    """Combine rankings without needing their scores to be comparable.

    RRF scores by 1/(k + rank), so a chunk that both retrievers rank decently
    beats one that a single retriever loves. That property is the whole reason
    hybrid search works: lexical and dense fail on *different* queries, and
    fusion keeps whichever one was right without you having to know in advance.
    """
    scores, seen = {}, {}
    for results in result_lists:
        for rank, (chunk, _score) in enumerate(results):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank + 1)
            seen[chunk.id] = chunk
    order = sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
    return [(seen[cid], s) for cid, s in order]


class HybridRetriever:
    """BM25 + optional dense, fused. Falls back to lexical alone."""

    def __init__(self, chunks, dense=None):
        self.bm25 = BM25(chunks)
        self.dense = dense

    def search(self, query, k=5, pool=20) -> list:
        lexical = self.bm25.search(query, k=pool)
        if self.dense is None:
            return lexical[:k]
        return reciprocal_rank_fusion([lexical, self.dense.search(query, k=pool)], top_k=k)


def rerank_by_overlap(query, results, top_k=5) -> list:
    """A stand-in for a cross-encoder reranker: term coverage of the query.

    A real reranker scores query and passage *jointly* and is the highest
    value-per-line component in most RAG systems — retrieve 50 cheaply, rerank
    to 5 precisely. This version demonstrates the shape (retrieve wide, score
    narrow) without a model download; swap in a cross-encoder for real work.
    """
    q = set(tokenize(query))
    scored = []
    for chunk, base in results:
        terms = set(tokenize(chunk.text))
        coverage = len(q & terms) / (len(q) or 1)
        scored.append((chunk, coverage + 0.001 * base))
    scored.sort(key=lambda cs: -cs[1])
    return scored[:top_k]


# --------------------------------------------------------------------------
# Retrieval metrics — measured separately from answer quality, always
# --------------------------------------------------------------------------


def recall_at_k(retrieved_ids, gold_ids, k=5) -> float:
    """Did the right passage make the cut? The ceiling on everything downstream.

    Measure this *before* you look at answer quality. If recall@5 is 0.6, the
    best imaginable model answers 40% of questions from nothing, and no amount
    of prompt engineering or fine-tuning will fix it.
    """
    gold = set(gold_ids)
    if not gold:
        return 1.0
    return len(gold & set(list(retrieved_ids)[:k])) / len(gold)


def mrr(retrieved_ids, gold_ids) -> float:
    """Mean reciprocal rank — position matters, because context windows are
    ordered and models attend unevenly across them."""
    gold = set(gold_ids)
    for i, cid in enumerate(retrieved_ids):
        if cid in gold:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved_ids, gold_ids, k=5) -> float:
    gold = set(gold_ids)
    dcg = sum((1.0 / math.log2(i + 2)) for i, cid in enumerate(list(retrieved_ids)[:k]) if cid in gold)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / ideal if ideal else 0.0


def evaluate_retrieval(retriever, cases, k=5) -> dict:
    """`cases` is [{"query", "gold_chunk_ids"}]. Returns the aggregate plus
    per-case rows, because the aggregate hides which queries fail."""
    rows = []
    for case in cases:
        hits = retriever.search(case["query"], k=k)
        ids = [c.id for c, _ in hits]
        rows.append({
            "query": case["query"],
            "retrieved": ids,
            "recall": recall_at_k(ids, case["gold_chunk_ids"], k),
            "mrr": mrr(ids, case["gold_chunk_ids"]),
            "ndcg": ndcg_at_k(ids, case["gold_chunk_ids"], k),
            "hit": recall_at_k(ids, case["gold_chunk_ids"], k) > 0,
        })
    n = len(rows) or 1
    return {
        "k": k,
        f"recall@{k}": sum(r["recall"] for r in rows) / n,
        "mrr": sum(r["mrr"] for r in rows) / n,
        f"ndcg@{k}": sum(r["ndcg"] for r in rows) / n,
        "hit_rate": sum(r["hit"] for r in rows) / n,
        "rows": rows,
    }


# --------------------------------------------------------------------------
# The triage
# --------------------------------------------------------------------------

RETRIEVAL_MISS = "retrieval_miss"
SYNTHESIS_MISS = "synthesis_miss"
CORRECT = "correct"


def triage(cases, k=5) -> dict:
    """Split failures into retrieval versus synthesis.

    `cases` is [{"query", "gold_chunk_ids", "retrieved_ids", "answer_correct"}].

    * gold passage absent from the top-k -> **retrieval miss**. The model was
      asked to recall something it was never shown. Fix chunking, embeddings,
      hybrid search, reranking. Days of work.
    * gold passage present, answer still wrong -> **synthesis miss**. Now a
      prompt, context-ordering, or model question. Weeks of work.

    The ratio is the recommendation. Nothing else in this repo converts an
    argument into a decision as quickly.
    """
    buckets = {RETRIEVAL_MISS: [], SYNTHESIS_MISS: [], CORRECT: []}
    for case in cases:
        retrieved = list(case.get("retrieved_ids", []))[:k]
        found = bool(set(case.get("gold_chunk_ids", [])) & set(retrieved))
        if case.get("answer_correct"):
            label = CORRECT
        else:
            label = SYNTHESIS_MISS if found else RETRIEVAL_MISS
        buckets[label].append(case)

    n = len(cases) or 1
    failures = len(buckets[RETRIEVAL_MISS]) + len(buckets[SYNTHESIS_MISS])
    return {
        "n": len(cases),
        "correct": len(buckets[CORRECT]),
        "retrieval_misses": len(buckets[RETRIEVAL_MISS]),
        "synthesis_misses": len(buckets[SYNTHESIS_MISS]),
        "failures": failures,
        "retrieval_share_of_failures": (len(buckets[RETRIEVAL_MISS]) / failures) if failures else 0.0,
        "accuracy": len(buckets[CORRECT]) / n,
        "buckets": buckets,
    }


def triage_recommendation(result) -> str:
    """Turn the split into the sentence you would say in the room."""
    r, s, f = result["retrieval_misses"], result["synthesis_misses"], result["failures"]
    if f == 0:
        return "No failures in this sample. Widen the eval set before concluding anything."
    share = result["retrieval_share_of_failures"]
    head = (f"{f} failures out of {result['n']}: {r} retrieval misses, {s} synthesis misses.")
    if share >= 0.6:
        return (f"{head}\n\nIn {r} of {f} cases the system was never shown the passage that "
                "contains the answer. That is a retrieval problem, not a model-intelligence "
                "problem — and it is the cheap one to fix. Fine-tuning would not have moved "
                f"any of those {r}. Recommendation: fix retrieval first (chunking, hybrid "
                "search, reranking), then re-run this same eval and compare the graph.")
    if share <= 0.3:
        return (f"{head}\n\nRetrieval is mostly doing its job — the right passage was in "
                f"context for {s} of the {f} failures and the answer was still wrong. That "
                "is a synthesis problem: prompt, context ordering, or model. Try prompt and "
                "ordering first; a model change is the expensive last resort, and fine-tuning "
                "is only right if the failures are style or format rather than reasoning.")
    return (f"{head}\n\nThe failures are split roughly evenly, so there is no single fix. "
            "Do retrieval first anyway — it is cheaper, and it changes the synthesis set "
            "you would be debugging afterwards.")


# --------------------------------------------------------------------------
# A corpus to practise on, so the lab runs offline
# --------------------------------------------------------------------------

POLICY_DOCS = {
    "refunds": """
    Refund policy. Customers may request a refund within 30 days of purchase for
    any unused subscription period. Refunds are issued to the original payment
    method within 5 to 10 business days. Annual plans cancelled after 30 days
    receive a prorated credit rather than a cash refund. Enterprise contracts are
    governed by their master services agreement and are not covered by the
    standard 30 day window. Refunds for chargebacks already in dispute are held
    until the dispute is resolved by the card network.
    """,
    "pricing": """
    Pricing. The Starter plan is 29 dollars per seat per month billed annually or
    35 dollars billed monthly. The Growth plan is 79 dollars per seat per month
    annually. Enterprise pricing is negotiated and includes a platform fee.
    Overage on API calls is billed at 40 cents per thousand calls above the
    included quota. Nonprofit and education customers receive a 40 percent
    discount on Starter and Growth. Price changes take effect at the next renewal
    and existing customers are notified 60 days in advance.
    """,
    "sla": """
    Service level agreement. The platform targets 99.9 percent monthly uptime for
    Growth customers and 99.95 percent for Enterprise. Scheduled maintenance is
    excluded from the calculation and is announced at least 72 hours ahead.
    Service credits are 10 percent of monthly fees for uptime below target, 25
    percent below 99 percent, and 50 percent below 95 percent. Credits must be
    requested within 30 days of the incident and are applied to the next invoice.
    """,
    "security": """
    Security and data handling. Data is encrypted in transit with TLS 1.3 and at
    rest with AES-256. Customer data is stored in the region selected at account
    creation and is not replicated across regions without written consent.
    Personnel access requires hardware key authentication and is logged. We retain
    audit logs for 400 days. Penetration tests are performed twice yearly and the
    summary report is available to Enterprise customers under NDA.
    """,
    "support": """
    Support. Starter includes email support with a 48 hour first response target.
    Growth includes 24 hour response and access to live chat during business
    hours. Enterprise includes a named technical account manager, a 4 hour
    response target for production incidents, and a dedicated escalation line.
    Severity one incidents are defined as a complete loss of production service
    and are handled continuously until resolved.
    """,
}

# Questions with the document that answers them. Deliberately mixed: some are
# keyword-shaped (BM25 wins), some are paraphrases (dense wins), and a few ask
# about facts that are *not* in the corpus at all — because a system that
# confidently answers those is the failure mode nobody tests for.
POLICY_QUESTIONS = [
    {"query": "How long do I have to request a refund?", "gold_doc": "refunds"},
    {"query": "Do annual plans get cash back if I cancel late?", "gold_doc": "refunds"},
    {"query": "What does the Growth plan cost per seat annually?", "gold_doc": "pricing"},
    {"query": "Is there a discount for nonprofits?", "gold_doc": "pricing"},
    {"query": "How much notice before prices go up?", "gold_doc": "pricing"},
    {"query": "What uptime do Enterprise customers get?", "gold_doc": "sla"},
    {"query": "How big is the service credit if uptime drops below 95 percent?", "gold_doc": "sla"},
    {"query": "Does planned downtime count against the SLA?", "gold_doc": "sla"},
    {"query": "What encryption is used for stored data?", "gold_doc": "security"},
    {"query": "How long are audit logs kept?", "gold_doc": "security"},
    {"query": "Can my data be copied to another region?", "gold_doc": "security"},
    {"query": "How fast does support answer a production outage on Enterprise?", "gold_doc": "support"},
    {"query": "What counts as a severity one incident?", "gold_doc": "support"},
    {"query": "Do I get a phone number to call?", "gold_doc": "support"},
    {"query": "What is the parental leave policy?", "gold_doc": None},
    {"query": "Which cloud provider hosts the platform?", "gold_doc": None},
]


def policy_corpus(chunk_size=60, overlap=15):
    """Chunks plus eval cases with gold chunk ids resolved."""
    chunks = chunk_documents(POLICY_DOCS, size=chunk_size, overlap=overlap)
    by_doc = {}
    for c in chunks:
        by_doc.setdefault(c.doc_id, []).append(c.id)
    cases = []
    for q in POLICY_QUESTIONS:
        cases.append({"query": q["query"],
                      "gold_doc": q["gold_doc"],
                      "gold_chunk_ids": by_doc.get(q["gold_doc"], []),
                      "answerable": q["gold_doc"] is not None})
    return chunks, cases
