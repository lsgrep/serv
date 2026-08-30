"""Show the work.

A function that returns `327680` teaches you to call a function. A function that
prints

    2 (K and V) x 80 layers x 8 kv_heads x 128 head_dim x 2 bytes
      = 327,680 B = 320.0 KiB per token

teaches you the formula, and the formula is the thing you need when someone
hands you a config.json you have never seen and asks you to size a deployment.

Nobody should memorise that a T4 has 320 GB/s of bandwidth or that Qwen2.5-3B
has 36 layers. Those are *given* to you — in a config file, on a spec sheet, by
the person asking. What you are being tested on is whether you can put them in
the right places. So every derivation here separates:

* **givens** — the numbers you were handed, and where each one is read from,
* **steps** — the substitution, written out, one line at a time,
* **checks** — a sanity test on the answer, because a formula applied
  confidently in the wrong units is worse than no answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def _fmt(v, unit=""):
    if v is None:
        return "(nothing derived yet)"
    if isinstance(v, str):
        return v
    if isinstance(v, float) and v != int(v):
        text = f"{v:,.4g}" if (abs(v) < 0.01 or abs(v) >= 1e6) else f"{v:,.2f}"
    else:
        text = f"{int(v):,}"
    return f"{text} {unit}".strip()


@dataclass
class Given:
    """An input, and where you would read it off."""

    name: str
    value: object
    unit: str = ""
    source: str = ""


def human_bytes(n) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:,.1f} TiB"  # pragma: no cover


@dataclass
class Step:
    """One substitution, shown rather than performed silently."""

    label: str
    expression: str
    value: object
    unit: str = ""
    note: str = ""


@dataclass
class Derivation:
    """A worked answer you could reproduce on a whiteboard."""

    title: str
    formula: str = ""
    givens: list = field(default_factory=list)
    steps: list = field(default_factory=list)
    checks: list = field(default_factory=list)
    result_label: str = "result"
    result_unit: str = ""
    warnings: list = field(default_factory=list)

    # -- building ---------------------------------------------------------
    def given(self, name, value, unit="", source=""):
        self.givens.append(Given(name, value, unit, source))
        return value

    def step(self, label, expression, value, unit="", note=""):
        self.steps.append(Step(label, expression, value, unit, note))
        return value

    def check(self, text):
        self.checks.append(text)
        return self

    def warn(self, text):
        self.warnings.append(text)
        return self

    # -- reading ----------------------------------------------------------
    @property
    def value(self):
        return self.steps[-1].value if self.steps else None

    @property
    def unit(self):
        return self.result_unit or (self.steps[-1].unit if self.steps else "")

    def __float__(self):
        return float(self.value)

    def __str__(self):
        width = 74
        out = [f"{self.title.upper()}", "=" * width]
        if self.formula:
            out += [f"  {self.formula}", ""]

        if self.givens:
            out.append("  GIVEN")
            name_w = max(len(g.name) for g in self.givens) + 2
            val_w = max(len(_fmt(g.value, g.unit)) for g in self.givens) + 2
            for g in self.givens:
                line = f"    {g.name:<{name_w}}{_fmt(g.value, g.unit):<{val_w}}"
                out.append(line + (f"  <- {g.source}" if g.source else ""))
            out.append("")

        if self.steps:
            out.append("  WORKING")
            for i, s in enumerate(self.steps, 1):
                out.append(f"    {i}. {s.label}")
                out.append(f"       {s.expression}")
                rendered = _fmt(s.value, s.unit)
                if s.unit == "B" and isinstance(s.value, (int, float)) and abs(s.value) >= 1024:
                    rendered += f"   ({human_bytes(s.value)})"
                out.append(f"       = {rendered}")
                if s.note:
                    out.append(f"       ({s.note})")
            out.append("")

        final = _fmt(self.value, self.unit)
        if self.unit == "B" and isinstance(self.value, (int, float)) and abs(self.value) >= 1024:
            final += f"   ({human_bytes(self.value)})"
        out.append(f"  {self.result_label.upper()}:  {final}")

        if self.checks:
            out.append("")
            out.append("  SANITY CHECK")
            for c in self.checks:
                out.append(f"    {c}")
        if self.warnings:
            out.append("")
            for w in self.warnings:
                out.append(f"  !! {w}")
        out.append("=" * width)
        return "\n".join(out)

    def _repr_pretty_(self, p, cycle):  # pragma: no cover - notebook display
        p.text(str(self))


class Worksheet:
    """Several derivations that feed each other, printed as one answer.

    This is the shape of a real sizing question: KV per token feeds the memory
    budget, which feeds concurrency; weights and bandwidth feed decode speed,
    which feeds cost. Each arrow is a place to be wrong, so each one is shown.
    """

    def __init__(self, title):
        self.title = title
        self.parts = []

    def add(self, derivation):
        self.parts.append(derivation)
        return derivation

    @property
    def results(self) -> dict:
        return {d.title: d.value for d in self.parts}

    def __str__(self):
        head = f"\n{'#' * 74}\n#  {self.title}\n{'#' * 74}\n"
        return head + "\n\n".join(str(p) for p in self.parts)

    def _repr_pretty_(self, p, cycle):  # pragma: no cover - notebook display
        p.text(str(self))
