"""Matplotlib helpers: a live serving dashboard, latency CDFs, sweep curves.

Design rules the charts here follow, because a plot you cannot read at a glance
during a debugging session is not worth drawing:

* Categorical hues are assigned in fixed order and never cycled — a series keeps
  its colour when another one is filtered out.
* One y-axis per plot. Two measures at different scales get two stacked axes
  sharing an x, never a twinned axis. (A dual-axis chart can be made to show any
  correlation you like by rescaling; that is why it is banned here.)
* Red is reserved for status — SLO breaches, errors, preemptions — never as
  "the fourth series".
* Every multi-series plot carries a legend, so identity is never colour alone.
"""

from __future__ import annotations

# Validated categorical palette: blue, orange, aqua, then yellow/magenta/green
# for the rare 4th+ series. Slots 1-3 are safe under colour-vision deficiency
# even when compared pairwise, which is what most of these charts need.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9"]
STATUS = {"good": "#008300", "warning": "#eda100", "serious": "#eb6834", "critical": "#e34948"}
GRID = "#d8d7d2"
INK = "#0b0b0b"
INK_MUTED = "#52514e"


def use_style(dark=False):
    """Apply the lab's chart defaults. Call once per notebook."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    surface = "#1a1a19" if dark else "#fcfcfb"
    ink = "#ffffff" if dark else INK
    muted = "#c3c2b7" if dark else INK_MUTED
    grid = "#383835" if dark else GRID
    mpl.rcParams.update({
        "figure.facecolor": surface,
        "axes.facecolor": surface,
        "savefig.facecolor": surface,
        "axes.edgecolor": grid,
        "axes.labelcolor": muted,
        "axes.titlecolor": ink,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": grid,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.7,
        "text.color": ink,
        "xtick.color": muted,
        "ytick.color": muted,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "legend.frameon": False,
        "figure.dpi": 110,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.prop_cycle": mpl.cycler(color=SERIES_DARK if dark else SERIES),
    })
    return plt


def _rows_to_cols(rows, keys):
    return {k: [r.get(k) for r in rows] for k in keys}


def dashboard(rows, title="serving under load", slo_ttft=None, ax_list=None):
    """The three-panel view: queue, memory, latency — sharing one time axis.

    Panel order is the order you read them in when something is wrong:
    *is work piling up*, *is memory the reason*, *what is the user feeling*.
    """
    import matplotlib.pyplot as plt

    cols = _rows_to_cols(rows, ["t", "running", "waiting", "kv_usage", "ttft_p50",
                                "ttft_p99", "preemptions", "gen_tokens_per_s"])
    t = cols["t"]
    if ax_list is None:
        fig, axes = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True,
                                 gridspec_kw={"hspace": 0.28})
    else:
        axes = ax_list
        fig = axes[0].figure

    a = axes[0]
    a.plot(t, cols["running"], color=SERIES[0], label="running")
    a.plot(t, cols["waiting"], color=SERIES[1], label="waiting (queued)")
    a.set_ylabel("requests")
    a.set_title(title)
    a.legend(loc="upper left", ncols=2)

    b = axes[1]
    b.plot(t, cols["kv_usage"], color=SERIES[2], label="KV cache used")
    b.set_ylabel("KV cache (%)")
    b.set_ylim(0, 105)
    if any(v for v in cols["preemptions"] if v):
        peak = max((v or 0) for v in cols["preemptions"])
        if peak:
            # Preemptions are an event count, not a rate — mark where they start
            # rather than plotting a second scale on this axis.
            first = next((tt for tt, v in zip(t, cols["preemptions"]) if v), None)
            if first is not None:
                b.axvline(first, color=STATUS["critical"], linestyle="--", linewidth=1.5)
                b.annotate(f"first preemption\n({peak:.0f} total)", xy=(first, 50),
                           xytext=(6, 0), textcoords="offset points",
                           color=STATUS["critical"], fontsize=9, va="center")
    b.legend(loc="upper left")

    c = axes[2]
    c.plot(t, [v * 1000 if v else None for v in cols["ttft_p50"]], color=SERIES[0], label="TTFT p50")
    c.plot(t, [v * 1000 if v else None for v in cols["ttft_p99"]], color=SERIES[1], label="TTFT p99")
    if slo_ttft:
        c.axhline(slo_ttft * 1000, color=STATUS["critical"], linestyle=":", linewidth=1.5)
        c.annotate("SLO", xy=(t[-1] if t else 0, slo_ttft * 1000), xytext=(-24, 4),
                   textcoords="offset points", color=STATUS["critical"], fontsize=9)
    c.set_ylabel("TTFT (ms)")
    c.set_xlabel("seconds since start")
    c.legend(loc="upper left", ncols=2)
    return fig, axes


def live_dashboard(poller, seconds=60, interval=1.0, slo_ttft=None, title="live"):
    """Redraw the dashboard in place while load runs. The notebook's Grafana.

    Run this in its own cell *after* kicking off the load generator in a
    background thread — watching the queue grow in real time is the intuition
    the whole lab is buying.
    """
    import time

    from IPython.display import clear_output

    plt = use_style()
    t0 = time.time()
    while time.time() - t0 < seconds:
        rows = list(poller.rows)
        if rows:
            clear_output(wait=True)
            fig, _ = dashboard(rows, title=f"{title} — t+{time.time() - t0:.0f}s", slo_ttft=slo_ttft)
            plt.show()
            last = rows[-1]
            print(f"running={last['running']:.0f}  waiting={last['waiting']:.0f}  "
                  f"kv={last['kv_usage']:.0f}%  preempt={last['preemptions']:.0f}  "
                  f"out_tok/s={last.get('gen_tokens_per_s', 0):.0f}")
        time.sleep(interval)


def latency_cdf(series, title="TTFT distribution", xlabel="TTFT (s)", slo=None, log=True):
    """CDF, not a histogram: percentiles are what SLOs are written in, and a CDF
    lets you read any percentile off one curve."""
    plt = use_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, (label, values) in enumerate(series.items()):
        vals = sorted(v for v in values if v is not None)
        if not vals:
            continue
        ys = [(j + 1) / len(vals) * 100 for j in range(len(vals))]
        ax.plot(vals, ys, label=label, color=SERIES[i % len(SERIES)])
    if slo:
        ax.axvline(slo, color=STATUS["critical"], linestyle=":", linewidth=1.5)
        ax.annotate("SLO", xy=(slo, 8), xytext=(4, 0), textcoords="offset points",
                    color=STATUS["critical"], fontsize=9)
    if log:
        ax.set_xscale("log")
    ax.set_ylim(0, 100)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("% of requests")
    ax.set_title(title)
    if len(series) > 1:
        ax.legend(loc="lower right")
    return fig, ax


def sweep_curves(rows, x="level", title="load sweep", xlabel="concurrency"):
    """Throughput and latency against offered load, stacked on a shared x.

    The shape to look for: throughput flattens while p99 keeps climbing. The
    knee is your operating point; everything to the right of it is latency you
    are paying for no extra work.
    """
    plt = use_style()
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={"hspace": 0.22})
    xs = [r[x] for r in rows]

    a = axes[0]
    a.plot(xs, [r["out_tok_per_s"] for r in rows], marker="o", color=SERIES[0], label="output tokens/s")
    if any(r.get("goodput") for r in rows):
        a.plot(xs, [r["req_per_s"] for r in rows], marker="o", color=SERIES[1], label="requests/s")
        a.plot(xs, [r["goodput"] for r in rows], marker="o", color=SERIES[2],
               label="goodput (req/s within SLO)")
    a.set_ylabel("throughput")
    a.set_title(title)
    a.legend(loc="upper left")

    b = axes[1]
    b.plot(xs, [(r["ttft_p50"] or 0) * 1000 for r in rows], marker="o", color=SERIES[0], label="TTFT p50")
    b.plot(xs, [(r["ttft_p99"] or 0) * 1000 for r in rows], marker="o", color=SERIES[1], label="TTFT p99")
    b.set_ylabel("latency (ms)")
    b.set_xlabel(xlabel)
    b.legend(loc="upper left")
    return fig, axes


def decode_curves(rows, title="cost of one decode step"):
    """Per-token latency vs context length, one line per cache setting."""
    plt = use_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    groups = {}
    for r in rows:
        groups.setdefault(r["cache"], []).append(r)
    for i, (label, rs) in enumerate(sorted(groups.items())):
        rs = sorted(rs, key=lambda r: r["context_len"])
        ax.plot([r["context_len"] for r in rs], [r["ms"] for r in rs],
                label=label, color=SERIES[i % len(SERIES)])
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("time per token (ms)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    return fig, ax


def bar_compare(labels, values, title="", ylabel="", highlight=None, fmt="{:,.0f}"):
    """Single-series bar chart with values labelled directly (no legend needed —
    the title names the measure)."""
    plt = use_style()
    fig, ax = plt.subplots(figsize=(max(4.5, 1.3 * len(labels)), 4))
    colors = [STATUS["critical"] if highlight and lbl in highlight else SERIES[0] for lbl in labels]
    bars = ax.bar(labels, values, color=colors, width=0.6)
    for rect, v in zip(bars, values):
        ax.annotate(fmt.format(v), xy=(rect.get_x() + rect.get_width() / 2, v),
                    xytext=(0, 3), textcoords="offset points", ha="center",
                    fontsize=9, color=INK_MUTED)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.margins(y=0.15)
    return fig, ax
