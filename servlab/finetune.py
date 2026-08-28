"""Datasets and scorers for the fine-tuning experiment.

The repo asserts, in two places, that **fine-tuning teaches style and format but
does not install knowledge**. That claim is load-bearing — it is the pivot of
the retrieval-versus-fine-tuning conversation — so it should be measured rather
than repeated.

This module builds the two datasets that let you measure it on one small model
in a few minutes:

* a **format** task, where the target behaviour is a strict output schema the
  base model does not reliably produce, and
* a **knowledge** task, where the target is facts that live in a corpus.

The knowledge task has a deliberate trap, and the trap is the experiment: the
training questions and the test questions ask about **the same facts in
different words**. A model that has memorised training phrasings scores well on
the first and poorly on the second. Retrieval scores well on both, because it
was handed the passage either way.

Everything here is pure Python — the datasets, the scorers, the splits — so the
argument is testable in CI and only the training loop needs a GPU.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Facts, with training and held-out phrasings kept strictly apart
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Fact:
    id: str
    doc: str
    answer: str
    train_questions: tuple
    heldout_questions: tuple
    accept: tuple = ()          # alternative strings that count as correct

    def accepted(self) -> tuple:
        return (self.answer,) + tuple(self.accept)


FACTS = (
    Fact("refund-window", "refunds", "30 days",
         ("How long do I have to request a refund?",
          "What is the refund window?",
          "How many days do I get to ask for a refund?"),
         ("If I change my mind about my purchase, by when must I say so?",
          "Is there a deadline for getting my money back?"),
         accept=("thirty days",)),
    Fact("refund-speed", "refunds", "5 to 10 business days",
         ("How long do refunds take to arrive?",
          "When will I see the refund on my card?"),
         ("After approval, how quickly does the money come back?",),
         accept=("5-10 business days", "five to ten business days")),
    Fact("growth-price", "pricing", "79 dollars",
         ("What does the Growth plan cost per seat annually?",
          "How much is Growth per seat?"),
         ("If I upgrade from Starter to the middle tier, what am I paying per person?",),
         accept=("$79", "79")),
    Fact("nonprofit-discount", "pricing", "40 percent",
         ("Is there a discount for nonprofits?",
          "What discount do education customers get?"),
         ("We're a registered charity — does that change the price?",),
         accept=("40%",)),
    Fact("price-notice", "pricing", "60 days",
         ("How much notice before prices go up?",
          "When are customers told about price changes?"),
         ("How far ahead do you warn people that billing will change?",),
         accept=("sixty days",)),
    Fact("enterprise-uptime", "sla", "99.95 percent",
         ("What uptime do Enterprise customers get?",
          "What is the Enterprise SLA target?"),
         ("For the top tier, how much downtime is contractually allowed?",),
         accept=("99.95%", "99.95")),
    Fact("credit-95", "sla", "50 percent",
         ("How big is the service credit if uptime drops below 95 percent?",
          "What credit applies under 95 percent uptime?"),
         ("If you have a really bad month — worse than 95 percent — what do we get back?",),
         accept=("50%",)),
    Fact("credit-window", "sla", "30 days",
         ("How long do I have to request a service credit?",),
         ("Is there a time limit on claiming compensation for an outage?",),
         accept=("thirty days",)),
    Fact("encryption-rest", "security", "AES-256",
         ("What encryption is used for stored data?",
          "How is data encrypted at rest?"),
         ("What protects the information once it is written to disk?",),
         accept=("aes 256",)),
    Fact("log-retention", "security", "400 days",
         ("How long are audit logs kept?",
          "What is the audit log retention period?"),
         ("For how long can you tell me who accessed my account?",),
         accept=("400",)),
    Fact("enterprise-response", "support", "4 hour",
         ("How fast does support answer a production outage on Enterprise?",
          "What is the Enterprise response target for production incidents?"),
         ("If our system is down and we're on the top plan, how soon do we hear back?",),
         accept=("four hour", "4 hours", "four hours")),
    Fact("sev1", "support", "complete loss of production service",
         ("What counts as a severity one incident?",),
         ("When is an issue treated as the most urgent category?",),
         accept=("total loss of production", "complete outage")),
)


# --------------------------------------------------------------------------
# The format task
# --------------------------------------------------------------------------

FORMAT_SYSTEM = (
    "You are a support assistant. Answer using only the provided context. "
    "Reply with a single JSON object and nothing else, with exactly these keys: "
    '"answer" (string), "citation" (the id of the passage you used), '
    '"confidence" (one of "high", "medium", "low").'
)

REQUIRED_KEYS = ("answer", "citation", "confidence")
CONFIDENCE_VALUES = ("high", "medium", "low")


@dataclass
class Example:
    prompt: str
    completion: str
    system: str = ""
    kind: str = ""                  # format | knowledge
    meta: dict = field(default_factory=dict)

    def as_messages(self) -> list:
        msgs = [{"role": "system", "content": self.system}] if self.system else []
        msgs.append({"role": "user", "content": self.prompt})
        msgs.append({"role": "assistant", "content": self.completion})
        return msgs


def _passage_for(doc_id, chunks):
    for c in chunks:
        if c.doc_id == doc_id:
            return c
    return None


def format_examples(chunks, split="train", seed=0) -> list:
    """Question + passage in, strict JSON out.

    This is what fine-tuning is genuinely good at: a house output format, applied
    at volume, that prompting alone gets right only most of the time. "Most of
    the time" is the problem — a parser downstream needs it every time, and
    retries cost latency and money.
    """
    rng = random.Random(seed)
    out = []
    for fact in FACTS:
        questions = fact.train_questions if split == "train" else fact.heldout_questions
        passage = _passage_for(fact.doc, chunks)
        if passage is None:
            continue
        for q in questions:
            completion = json.dumps({
                "answer": fact.answer,
                "citation": passage.id,
                "confidence": rng.choice(["high", "high", "medium"]),
            })
            out.append(Example(
                prompt=f"Context [{passage.id}]:\n{passage.text}\n\nQuestion: {q}",
                completion=completion, system=FORMAT_SYSTEM, kind="format",
                meta={"fact": fact.id, "citation": passage.id},
            ))
    rng.shuffle(out)
    return out


KNOWLEDGE_SYSTEM = ("You are a support assistant. Answer the question directly and "
                    "briefly, from what you know about this company's policies.")


def knowledge_examples(split="train", seed=0) -> list:
    """Closed-book question and answer pairs — no passage in context.

    Training on these is the thing people *mean* when they say "fine-tune it on
    our docs". The held-out split asks about the same facts in different words,
    which is the only version of the test that tells you anything: scoring well
    on phrasings you trained on measures memorisation, not knowledge.
    """
    rng = random.Random(seed)
    out = []
    for fact in FACTS:
        questions = fact.train_questions if split == "train" else fact.heldout_questions
        for q in questions:
            out.append(Example(prompt=q, completion=fact.answer, system=KNOWLEDGE_SYSTEM,
                               kind="knowledge",
                               meta={"fact": fact.id, "accept": fact.accepted(),
                                     "doc": fact.doc}))
    rng.shuffle(out)
    return out


def augment(examples, factor=8, seed=0) -> list:
    """Repeat the set, because twelve facts is not a training run.

    Deliberately naive: real augmentation would paraphrase. Repetition is what
    a team under deadline actually does, and it is worth seeing what it buys —
    which is memorisation of these exact strings, and that is the point.
    """
    rng = random.Random(seed)
    out = [Example(e.prompt, e.completion, e.system, e.kind, dict(e.meta))
           for _ in range(factor) for e in examples]
    rng.shuffle(out)
    return out


def to_text(example, template="chatml") -> str:
    """Render to a training string when no tokenizer chat template is available."""
    if template == "chatml":
        parts = []
        if example.system:
            parts.append(f"<|im_start|>system\n{example.system}<|im_end|>")
        parts.append(f"<|im_start|>user\n{example.prompt}<|im_end|>")
        parts.append(f"<|im_start|>assistant\n{example.completion}<|im_end|>")
        return "\n".join(parts)
    return f"### System\n{example.system}\n\n### User\n{example.prompt}\n\n### Assistant\n{example.completion}"


def to_dataset(examples, tokenizer=None) -> list:
    """[{"text": ...}] rows, using the tokenizer's own chat template when given —
    a mismatched template is the quiet reason a fine-tune underperforms."""
    rows = []
    for e in examples:
        if tokenizer is not None and getattr(tokenizer, "chat_template", None):
            text = tokenizer.apply_chat_template(e.as_messages(), tokenize=False)
        else:
            text = to_text(e)
        rows.append({"text": text})
    return rows


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def score_format(output, valid_citations=()) -> dict:
    """Did it produce the schema — strictly, and with nothing around it?

    `strict` is the number that matters. A downstream parser does not care that
    the JSON was in there somewhere; it cares whether the whole response is the
    object. Reporting `parses` alone flatters the model.
    """
    text = (output or "").strip()
    result = {"strict": False, "parses": False, "keys_ok": False,
              "confidence_ok": False, "citation_valid": False, "extra_prose": False}
    obj = None
    try:
        obj = json.loads(text)
        result["strict"] = isinstance(obj, dict)
        result["parses"] = result["strict"]
    except (json.JSONDecodeError, TypeError):
        m = _JSON_BLOCK.search(text)
        if m:
            try:
                obj = json.loads(m.group(0))
                result["parses"] = isinstance(obj, dict)
                result["extra_prose"] = True
            except json.JSONDecodeError:
                obj = None
    if isinstance(obj, dict):
        result["keys_ok"] = all(k in obj for k in REQUIRED_KEYS)
        result["confidence_ok"] = str(obj.get("confidence", "")).lower() in CONFIDENCE_VALUES
        result["citation_valid"] = (not valid_citations) or (obj.get("citation") in set(valid_citations))
    return result


def format_rate(outputs, valid_citations=()) -> dict:
    """Aggregate the format scores. `usable` is strict JSON with the right keys —
    the only column a downstream system cares about."""
    scores = [score_format(o, valid_citations) for o in outputs]
    n = len(scores) or 1
    usable = sum(1 for s in scores if s["strict"] and s["keys_ok"])
    return {
        "n": len(scores),
        "usable": usable / n,
        "parses_anywhere": sum(s["parses"] for s in scores) / n,
        "strict_json": sum(s["strict"] for s in scores) / n,
        "keys_ok": sum(s["keys_ok"] for s in scores) / n,
        "confidence_ok": sum(s["confidence_ok"] for s in scores) / n,
        "citation_valid": sum(s["citation_valid"] for s in scores) / n,
        "wrapped_in_prose": sum(s["extra_prose"] for s in scores) / n,
    }


def _normalise(text) -> str:
    return re.sub(r"[^a-z0-9%. ]+", " ", (text or "").lower())


def score_knowledge(output, example) -> bool:
    """Lenient on wording, strict on the fact.

    A knowledge eval that demands exact strings measures formatting; one that
    accepts any answer containing a number measures nothing. Accepting a small
    set of equivalent surface forms is the compromise, and it should be visible
    in the data rather than buried in the scorer.
    """
    out = _normalise(output)
    for candidate in example.meta.get("accept", (example.completion,)):
        if _normalise(candidate).strip() in out:
            return True
    return False


def evaluate_knowledge(examples, generate, name="model") -> dict:
    """`generate(prompt, system) -> str`, so the same eval runs closed-book,
    few-shot, or with retrieved context pasted in."""
    rows = []
    for e in examples:
        out = generate(e.prompt, e.system)
        rows.append({"fact": e.meta.get("fact"), "question": e.prompt,
                     "expected": e.completion, "got": (out or "").strip()[:160],
                     "correct": score_knowledge(out, e)})
    n = len(rows) or 1
    return {"name": name, "n": len(rows),
            "accuracy": sum(r["correct"] for r in rows) / n, "rows": rows}


# --------------------------------------------------------------------------
# The regression check nobody runs
# --------------------------------------------------------------------------

CAPABILITY_PROBES = (
    ("What is 17 plus 26?", "43"),
    ("Name the capital city of Japan.", "tokyo"),
    ("Translate to French: good morning", "bonjour"),
    ("What comes next: 2, 4, 8, 16,", "32"),
    ("Write the word 'serving' backwards.", "gnivres"),
    ("Is 91 a prime number? Answer yes or no.", "no"),
    ("How many days are in a leap year?", "366"),
    ("What is the chemical symbol for gold?", "au"),
)


def capability_check(generate, probes=CAPABILITY_PROBES, name="model") -> dict:
    """Did the fine-tune damage anything it was not supposed to touch?

    Narrow fine-tuning on a repetitive format is a very effective way to teach a
    small model to emit that format *regardless of the question*. Checking for it
    takes two minutes and almost nobody does — which is why "it works great on
    our task" and "it broke everything else" so often ship together.
    """
    rows = []
    for prompt, expected in probes:
        out = generate(prompt, "")
        rows.append({"prompt": prompt, "expected": expected,
                     "got": (out or "").strip()[:120],
                     "correct": _normalise(expected).strip() in _normalise(out)})
    n = len(rows) or 1
    return {"name": name, "n": len(rows),
            "accuracy": sum(r["correct"] for r in rows) / n, "rows": rows}


# --------------------------------------------------------------------------
# Presentation
# --------------------------------------------------------------------------


def comparison_table(results) -> str:
    """`results` is [{"name", "format_usable", "known_phrasings", "held_out",
    "with_retrieval", "capability"}] — any missing column prints as a dash."""
    cols = [("name", "system", 26), ("format_usable", "format", 9),
            ("known_phrasings", "seen Qs", 9), ("held_out", "held-out", 10),
            ("with_retrieval", "+ RAG", 8), ("capability", "general", 9)]
    head = "".join(f"{label:<{w}}" if key == "name" else f"{label:>{w}}"
                   for key, label, w in cols)
    lines = [head, "-" * len(head)]
    for r in results:
        row = ""
        for key, _label, w in cols:
            v = r.get(key)
            if key == "name":
                row += f"{str(v):<{w}}"
            elif v is None:
                row += f"{'-':>{w}}"
            else:
                row += f"{v:>{w - 1}.0%} "
        lines.append(row)
    return "\n".join(lines)


def verdict(fine_tuned_heldout, base_heldout, rag_heldout, format_gain) -> str:
    """The sentence the experiment earns."""
    knowledge_gain = fine_tuned_heldout - base_heldout
    lines = []
    lines.append(f"format adherence moved {format_gain:+.0%} — fine-tuning is very good at this")
    lines.append(f"held-out factual recall moved {knowledge_gain:+.0%} from fine-tuning alone")
    lines.append(f"the same base model with retrieval scores {rag_heldout:.0%} on those questions")
    lines.append("")
    if knowledge_gain < 0.15 and rag_heldout > fine_tuned_heldout:
        lines.append(
            "Conclusion, measured rather than asserted: the fine-tune learned the output "
            "format and the training phrasings. Asked the same facts in different words it "
            "is barely better than the base model, while retrieval — which was handed the "
            "passage — answers them. Fine-tuning taught behaviour; it did not install "
            "knowledge.")
    elif knowledge_gain >= 0.15:
        lines.append(
            "Held-out recall genuinely improved, which happens when the facts are dense, "
            "repeated, and consistent with what the base model already half-knew. Note it "
            "still needs the eval to prove it, and that retrieval got there without "
            "training. Report both numbers.")
    else:
        lines.append("Inconclusive at this sample size — widen the eval before drawing a "
                     "conclusion either way.")
    return "\n".join(lines)
