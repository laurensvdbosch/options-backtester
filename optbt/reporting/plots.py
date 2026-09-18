"""Matplotlib charts. All functions save a PNG and return the path.

Colour use is deliberate and fixed: frictionless is always blue, realistic is
always orange, market/spot is aqua and implied vol is yellow. Every chart has
a single y-axis; related measures with different scales get their own panel.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from optbt.analytics.metrics import drawdown_series  # noqa: E402
from optbt.engine import BacktestResult  # noqa: E402

COLOR_FRICTIONLESS = "#2a78d6"
COLOR_REALISTIC = "#eb6834"
COLOR_SPOT = "#1baf7a"
COLOR_IV = "#eda100"
COLOR_GRID = "#d9d9d6"
COLOR_TEXT = "#3a3a38"

plt.rcParams.update(
    {
        "axes.edgecolor": COLOR_GRID,
        "axes.grid": True,
        "grid.color": COLOR_GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": COLOR_TEXT,
        "xtick.color": COLOR_TEXT,
        "ytick.color": COLOR_TEXT,
        "text.color": COLOR_TEXT,
        "font.size": 9,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
        "figure.dpi": 120,
    }
)


def plot_equity_comparison(pairs: dict[str, tuple[BacktestResult, BacktestResult]], out: str | Path) -> Path:
    n = len(pairs)
    fig, axes = plt.subplots(2, n, figsize=(5.2 * n, 6.4), sharex="col", squeeze=False)
    for col, (name, (f, r)) in enumerate(pairs.items()):
        ax = axes[0][col]
        ax.plot(f.equity.index, f.equity["nav"], color=COLOR_FRICTIONLESS, label="frictionless")
        ax.plot(r.equity.index, r.equity["nav"], color=COLOR_REALISTIC, label="realistic costs")
        ax.set_title(name, loc="left", fontsize=10)
        if col == 0:
            ax.set_ylabel("NAV ($)")
            ax.legend(loc="upper left")
        ax2 = axes[1][col]
        ax2.fill_between(f.equity.index, drawdown_series(f.equity["nav"]) * 100, 0, color=COLOR_FRICTIONLESS, alpha=0.35, linewidth=0)
        ax2.fill_between(r.equity.index, drawdown_series(r.equity["nav"]) * 100, 0, color=COLOR_REALISTIC, alpha=0.35, linewidth=0)
        if col == 0:
            ax2.set_ylabel("Drawdown (%)")
        ax2.tick_params(axis="x", labelrotation=30)
    fig.suptitle("Equity curves and drawdowns: frictionless vs realistic execution", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    out = Path(out)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_greeks(result: BacktestResult, out: str | Path) -> Path:
    eq = result.equity
    panels = [
        ("delta", "Delta (share-equiv.)"),
        ("gamma", "Gamma (delta per $1)"),
        ("theta", "Theta ($/day)"),
        ("vega", "Vega ($/vol pt)"),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 8.5), sharex=True)
    for ax, (col, label) in zip(axes, panels):
        ax.plot(eq.index, eq[col], color=COLOR_REALISTIC if result.cost_label == "realistic" else COLOR_FRICTIONLESS)
        ax.axhline(0, color=COLOR_TEXT, linewidth=0.6, alpha=0.5)
        ax.set_ylabel(label)
    axes[-1].tick_params(axis="x", labelrotation=30)
    fig.suptitle(f"Portfolio Greek exposure: {result.strategy_name} ({result.cost_label})", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    out = Path(out)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_market(result: BacktestResult, out: str | Path) -> Path:
    eq = result.equity
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
    axes[0].plot(eq.index, eq["spot"], color=COLOR_SPOT)
    axes[0].set_ylabel("Spot")
    axes[1].plot(eq.index, eq["atm_iv"] * 100, color=COLOR_IV)
    axes[1].set_ylabel("ATM 30d implied vol (%)")
    axes[1].tick_params(axis="x", labelrotation=30)
    fig.suptitle("Simulated underlying and implied-vol level", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    out = Path(out)
    fig.savefig(out)
    plt.close(fig)
    return out
