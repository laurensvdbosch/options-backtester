"""Markdown / plain-text reporting.

The central artefact is :func:`build_report`, which takes ``{strategy_name:
(frictionless_result, realistic_result)}`` pairs and produces a Markdown
document with side-by-side metrics, a cost decomposition and Greek-exposure
tables. Every table function also works standalone.
"""

from __future__ import annotations

import math
from typing import Iterable

from optbt.engine import BacktestResult

_METRIC_ROWS: list[tuple[str, str, str]] = [
    # (key, label, format)
    ("total_pnl", "Total P&L ($)", "money"),
    ("total_return", "Total return", "pct"),
    ("cagr", "CAGR", "pct"),
    ("ann_volatility", "Ann. volatility", "pct"),
    ("sharpe", "Sharpe", "num2"),
    ("sortino", "Sortino", "num2"),
    ("max_drawdown", "Max drawdown", "pct"),
    ("max_drawdown_days", "Max DD duration (days)", "int"),
    ("calmar", "Calmar", "num2"),
    ("n_trades", "Closed trades", "int"),
    ("win_rate", "Win rate", "pct"),
    ("profit_factor", "Profit factor", "num2"),
    ("expectancy", "Expectancy / trade ($)", "money"),
]


def fmt(value: float | int | None, kind: str = "num2") -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value))):
        return "n/a"
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    if kind == "pct":
        return f"{value:.2%}"
    if kind == "money":
        return f"{value:,.0f}"
    if kind == "int":
        return f"{int(value):,d}"
    if kind == "num1":
        return f"{value:,.1f}"
    return f"{value:,.2f}"


def _md_table(header: list[str], rows: Iterable[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def metrics_table(results: list[BacktestResult]) -> str:
    header = ["Metric"] + [f"{r.strategy_name} ({r.cost_label})" for r in results]
    rows = []
    for key, label, kind in _METRIC_ROWS:
        rows.append([label] + [fmt(r.metrics.get(key), kind) for r in results])
    return _md_table(header, rows)


def comparison_table(pairs: dict[str, tuple[BacktestResult, BacktestResult]]) -> str:
    """One row per strategy: frictionless vs realistic headline numbers."""
    header = [
        "Strategy",
        "P&L frictionless",
        "P&L realistic",
        "Cost drag",
        "Explicit costs",
        "Sharpe (f)",
        "Sharpe (r)",
        "Max DD (f)",
        "Max DD (r)",
        "Win rate (f)",
        "Win rate (r)",
    ]
    rows = []
    for name, (f, r) in pairs.items():
        rows.append(
            [
                name,
                fmt(f.total_pnl, "money"),
                fmt(r.total_pnl, "money"),
                fmt(r.total_pnl - f.total_pnl, "money"),
                fmt(-r.cost_summary["total"], "money"),
                fmt(f.metrics["sharpe"]),
                fmt(r.metrics["sharpe"]),
                fmt(f.metrics["max_drawdown"], "pct"),
                fmt(r.metrics["max_drawdown"], "pct"),
                fmt(f.metrics["win_rate"], "pct"),
                fmt(r.metrics["win_rate"], "pct"),
            ]
        )
    return _md_table(header, rows)


def cost_table(results: list[BacktestResult]) -> str:
    header = ["Strategy (costs)", "Fills", "Contracts", "Spread ($)", "Slippage ($)", "Commission ($)", "Total ($)", "% of start NAV"]
    rows = []
    for r in results:
        c = r.cost_summary
        rows.append(
            [
                f"{r.strategy_name} ({r.cost_label})",
                fmt(c["n_fills"], "int"),
                fmt(c["contracts_traded"], "int"),
                fmt(c["spread"], "money"),
                fmt(c["slippage"], "money"),
                fmt(c["commission"], "money"),
                fmt(c["total"], "money"),
                fmt(c["total"] / r.initial_cash, "pct"),
            ]
        )
    return _md_table(header, rows)


def greeks_table(results: list[BacktestResult]) -> str:
    header = ["Strategy (costs)", "Delta mean", "Delta min", "Delta max", "Gamma mean", "Theta mean ($/day)", "Vega mean ($/pt)", "Vega min", "Days short vega"]
    rows = []
    for r in results:
        g = r.greek_summary
        rows.append(
            [
                f"{r.strategy_name} ({r.cost_label})",
                fmt(g["delta"]["mean"], "num1"),
                fmt(g["delta"]["min"], "num1"),
                fmt(g["delta"]["max"], "num1"),
                fmt(g["gamma"]["mean"], "num2"),
                fmt(g["theta"]["mean"], "num1"),
                fmt(g["vega"]["mean"], "num1"),
                fmt(g["vega"]["min"], "num1"),
                fmt(g["pct_days_short_vega"], "pct"),
            ]
        )
    return _md_table(header, rows)


def market_summary(result: BacktestResult) -> str:
    m = result.config.market
    p = result.path
    eq = result.equity
    lines = [
        f"- Model: **{m.model}**, seed {m.seed}, {len(p.dates)} trading days ({p.dates[0]} to {p.dates[-1]})",
        f"- Spot: start {p.spot[0]:.2f}, end {p.spot[-1]:.2f}, min {p.spot.min():.2f}, max {p.spot.max():.2f}",
        f"- Realised vol of the path: {p.realized_vol:.1%}; ATM 30d IV mean {eq['atm_iv'].mean():.1%} (min {eq['atm_iv'].min():.1%}, max {eq['atm_iv'].max():.1%})",
        f"- Surface: skew {m.skew:+.3f}, smile {m.smile:+.3f}, IV premium over model vol {m.iv_premium:.0%}",
        f"- Rates: r = {m.r:.2%}, q = {m.q:.2%}; settlement: {result.config.execution.settlement}",
    ]
    return "\n".join(lines)


def build_report(pairs: dict[str, tuple[BacktestResult, BacktestResult]], title: str = "Options backtest report") -> str:
    all_results = [r for pair in pairs.values() for r in pair]
    first = all_results[0]
    e = first.config.execution
    sections = [
        f"# {title}",
        "",
        "## Market",
        market_summary(first),
        "",
        "## Strategies",
        "\n".join(f"- **{name}**: {pair[0].strategy_description}" for name, pair in pairs.items()),
        "",
        "## Frictionless vs realistic execution",
        "Same simulated path and vol surface; the only difference is the execution model. "
        f"Realistic run: spread = max(${e.spread_min_abs:.2f}, {e.spread_base_rel:.0%}-{e.spread_base_rel + e.spread_otm_rel:.0%} of extrinsic "
        f"+ {e.spread_intrinsic_rel:.1%} of intrinsic), slippage impact {e.slippage_impact} x sqrt(size/depth), "
        f"commission ${e.commission_per_contract:.2f}/contract.",
        "",
        comparison_table(pairs),
        "",
        "*Cost drag* is the P&L difference between the two runs; *explicit costs* is the sum of half-spread, slippage "
        "and commission actually paid in the realistic run. They differ when costs change the path of the strategy "
        "(a stop or profit target triggering on a different day).",
        "",
        "## Cost decomposition",
        cost_table(all_results),
        "",
        "## Full metrics",
        metrics_table(all_results),
        "",
        "## Greek exposure (daily portfolio aggregates)",
        greeks_table(all_results),
        "",
        "Delta is in share-equivalents, theta in $/calendar day, vega in $ per vol point.",
    ]
    return "\n".join(sections) + "\n"
