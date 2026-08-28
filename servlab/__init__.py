"""servlab — a lab kit for learning LLM inference serving by measuring it.

The modules are deliberately split by what they need:

    napkin, prometheus, stats, toy.allocator, toy.scheduler   pure python
    monitor, loadgen, serve, evalkit                          network only
    toy.engine, memory                                        need torch/GPU

so the arithmetic and the scheduling policy can be tested on a CPU runner, and
only the cells that touch a model need an accelerator.
"""

__version__ = "0.1.0"

from . import napkin, prometheus, stats  # noqa: F401

__all__ = ["napkin", "prometheus", "stats", "__version__"]


def notebook_setup(dark=False, style=True):
    """One call for cell 1 of a notebook: print the environment, set chart style.

    Returns the detected `Env` so the notebook can branch on `env.supports_bf16`
    and friends instead of hard-coding a card.
    """
    from .env import banner

    env = banner()
    if style:
        try:
            from .plots import use_style

            use_style(dark=dark)
        except ImportError:
            print("(matplotlib not installed yet — charts will be unstyled)")
    return env
