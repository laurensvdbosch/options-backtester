"""Run the three example strategies with and without realistic execution costs.

Usage (from the repository root)::

    python examples/run_comparison.py                 # Heston market, 2 years, seed 42
    python examples/run_comparison.py --model gbm --seed 7 --years 3
    python examples/run_comparison.py --out my_output

Outputs a Markdown report, CSV logs and PNG charts in ``--out`` (default ``output/``).
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optbt import Backtester, BacktestConfig, ExecutionConfig, MarketConfig  # noqa: E402
from optbt.reporting import build_report, plot_equity_comparison, plot_greeks, plot_market  # noqa: E402
from optbt.strategy import BullPutSpread, CoveredCall, ShortCall, ShortStrangle  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["heston", "gbm"], default="heston")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--years", type=float, default=2.0)
    p.add_argument("--start", type=date.fromisoformat, default=date(2022, 1, 3))
    p.add_argument("--cash", type=float, default=100_000.0)
    p.add_argument("--out", type=Path, default=Path("output"))
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    end = date(args.start.year + int(args.years), args.start.month, args.start.day)
    if args.years % 1:
        end = date.fromordinal(args.start.toordinal() + int(args.years * 365))
    market = MarketConfig(model=args.model, seed=args.seed, start=args.start, end=end)

    # Strategy factories: a fresh instance per run so internal counters start clean.
    # Covered calls are physically settled (shares get called away); the defined-risk
    # spreads are cash settled, as index options would be.
    specs = [
        ("covered_call", lambda: CoveredCall(lots=5), "physical"),
        ("bull_put_spread", lambda: BullPutSpread(contracts=10), "cash"),
        ("short_strangle", lambda: ShortStrangle(contracts=5), "cash"),
        ("short_call_20d", lambda: ShortCall(contracts=5, target_delta=0.20, take_profit=0.5), "cash"),
        ("covered_call_20d", lambda: CoveredCall(lots=5, target_delta=0.20, take_profit=0.5), "physical"),
    ]

    pairs = {}
    t0 = time.time()
    for name, factory, settlement in specs:
        results = []
        for frictionless in (True, False):
            execution = ExecutionConfig(initial_cash=args.cash, settlement=settlement, frictionless=frictionless)
            bt = Backtester(factory(), BacktestConfig(market=market, execution=execution))
            res = bt.run()
            results.append(res)
            print(
                f"{name:16s} {res.cost_label:12s} P&L {res.total_pnl:>10,.0f}  Sharpe {res.metrics['sharpe']:6.2f}  "
                f"MaxDD {res.metrics['max_drawdown']:7.2%}  trades {res.metrics['n_trades']:3d}  "
                f"costs {res.cost_summary['total']:8,.0f}"
            )
        pairs[name] = (results[0], results[1])
    print(f"\n{len(specs) * 2} backtests in {time.time() - t0:.1f}s")

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    report = build_report(pairs)
    (out / "report.md").write_text(report, encoding="utf-8")
    for name, (f, r) in pairs.items():
        f.equity.to_csv(out / f"{name}_frictionless_equity.csv")
        r.equity.to_csv(out / f"{name}_realistic_equity.csv")
        r.fills.to_csv(out / f"{name}_realistic_fills.csv", index=False)
        r.trades.to_csv(out / f"{name}_realistic_trades.csv", index=False)
        r.settlements.to_csv(out / f"{name}_realistic_settlements.csv", index=False)
    if not args.no_plots:
        plot_equity_comparison(pairs, out / "equity_comparison.png")
        plot_market(next(iter(pairs.values()))[1], out / "market.png")
        for name, (_, r) in pairs.items():
            plot_greeks(r, out / f"{name}_greeks.png")

    print("\n" + report)
    print(f"Report and charts written to {out.resolve()}")


if __name__ == "__main__":
    main()
