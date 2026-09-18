# Plain-English guide to this project

The README is the technical document. This file is the friendly one: where things are, how to change them, and how to talk about the project.

---

## 1. What this thing actually does

It pretends to be a stock market for two years, one day at a time. Each day it:

1. Moves the stock price a little (randomly, but in a realistic way).
2. Works out prices for every call and put option on that stock.
3. Asks your strategy "do you want to buy or sell anything today?"
4. Fills your orders, charging you the spread, slippage and commission like a real broker would.
5. Settles any options that expire that day.
6. Writes down your account value and your risk numbers.

At the end it tells you how the strategy did, and it runs everything twice: once with trading costs switched off, once with them on. That comparison is the whole point of the project.

---

## 2. Where things live

```
options-backtester/
├── GUIDE.md                 <- you are here
├── README.md                <- the full technical write-up
├── examples/run_comparison.py   <- THE FILE YOU RUN. Lists which strategies to test.
├── optbt/                   <- the engine (you rarely need to touch this)
│   ├── strategy/            <- THE FOLDER YOU EDIT. One file per strategy.
│   ├── config.py            <- market and cost settings (starting cash, volatility, commissions...)
│   ├── market/              <- fake stock prices and option prices
│   ├── execution/           <- the fake broker: fills, costs, expiry
│   ├── analytics/           <- Sharpe, drawdown, win rate, Greeks
│   └── reporting/           <- the report and the charts
├── tests/                   <- automated checks that the maths is right
└── output/                  <- created when you run it: report.md, charts, spreadsheets
```

If you only ever open two things, make them `examples/run_comparison.py` and the `optbt/strategy/` folder.

---

## 3. Running it

Open a terminal in this folder (in File Explorer: right-click inside the folder, "Open in Terminal") and type:

```
python examples/run_comparison.py
```

Wait about 30 seconds. Then open the `output` folder:

- `report.md` is the results table. Open it in any text editor, or in VS Code for nice formatting.
- `equity_comparison.png` shows the account value over time, with and without costs.
- `*_greeks.png` shows your risk exposure over time for each strategy.
- The `.csv` files open in Excel: every trade, every fill, every expiry.

Useful variations:

```
python examples/run_comparison.py --seed 7          # a different random market
python examples/run_comparison.py --years 5         # a longer test
python examples/run_comparison.py --model gbm       # simpler price model, no vol clustering
```

Try several seeds. One run is one possible history; a strategy that only works on seed 42 does not work.

---

## 4. Changing things without writing new code

**Which strategies run, and with what settings.** Open `examples/run_comparison.py` and find the list called `specs`. Each line is one strategy. For example:

```python
("short_call_20d", lambda: ShortCall(contracts=5, target_delta=0.20, take_profit=0.5), "cash"),
```

Change `0.20` to `0.30` for a 30-delta call, `0.5` to `0.7` for a 70% profit target, `5` to `1` for one contract. Delete a line to skip that strategy. The last word (`"cash"` or `"physical"`) is how options settle at expiry: physical means shares change hands, cash means you just get the money difference.

**Market and cost assumptions.** Open `optbt/config.py`. `MarketConfig` holds starting price, interest rate, how volatile the stock is, how skewed option prices are. `ExecutionConfig` holds starting cash, commission per contract, how wide the bid-ask spread is. Every setting has a comment next to it.

---

## 5. Adding your own strategy (when you're ready)

Copy `optbt/strategy/short_call.py`, rename it, and change three things:

1. `class ShortCall` -> `class YourName`, and `name = "short_call"` -> `name = "your_name"`.
2. The rules inside `on_bar`. This method runs once per day and returns a list of orders.
3. Add one line to `optbt/strategy/__init__.py` so it can be imported, and one line to the `specs` list in `run_comparison.py`.

Inside `on_bar` you have a variable `ctx` (context) that answers questions about today:

| You want to know... | Use |
|---|---|
| a call or put with a certain delta, expiring in 30 to 45 days | `ctx.chain.find(OptionType.CALL, target_delta=0.30, min_dte=30, max_dte=45)` |
| the same but by strike | `ctx.chain.find(OptionType.PUT, strike=95, min_dte=30, max_dte=45)` |
| what options I currently hold | `ctx.option_positions()` (each has `.quantity`, `.avg_price`, `.instrument`) |
| how many shares I hold | `ctx.stock_quantity()` |
| today's price of something I hold | `ctx.quote(position.instrument)` then `.bid`, `.ask`, `.mid`, `.delta`, `.dte` |
| an order that closes a position | `ctx.close_order(position)` |
| my overall delta / vega right now | `ctx.greeks["delta"]`, `ctx.greeks["vega"]` |

And you return orders like `Order(quote.contract, Side.SELL, 1, tag=self.next_tag())`. Give both legs of a spread the same tag so the stats count them as one trade.

Good next strategies to try, roughly in order of difficulty: a protective put, an iron condor (four legs), a calendar spread (two expiries), and a delta-hedged straddle that uses `ctx.greeks["delta"]` to trade shares against the options.

Run `python -m pytest` after changes to the engine itself. It takes ten seconds and tells you if you broke the maths. Strategy changes don't need it.

---

## 6. Putting it on your CV

Be honest about how it was made: it was built with an AI coding assistant to your specification, and the value you bring is understanding what it does and why. Interviewers will ask, so make sure you can explain these in your own words:

- Why simulate prices instead of using real option data, and what that costs you in realism.
- Why fills are not at the mid price, and why far out-of-the-money options have the widest spreads in percentage terms.
- What Sharpe, max drawdown and win rate each tell you, and why a 97% win rate can still be a bad strategy.
- What delta and vega mean for a short strangle, and what the Greeks chart shows.
- Why the bull put spread went from profitable to losing once costs were added.

A CV line that fits on one or two bullets:

> **Options strategy backtester (Python)** — Built a modular backtesting engine that simulates an underlying (GBM / Heston stochastic volatility), prices a full option chain daily with Black-Scholes and a skewed implied-vol surface, and models realistic execution (bid-ask spread, liquidity-based slippage, commissions, cash/physical expiry settlement). Reports Sharpe, Sortino, drawdown, win rate and daily Greek exposure; 100+ unit tests. Used it to show that a short put spread's edge is fully consumed by transaction costs.

Good places to mention it: a "Projects" section on the CV, a GitHub repository (the `.gitignore` is already set up, and the charts in `docs/` will show on the GitHub page), and a LinkedIn post with the equity chart and one sentence about what costs did to the spread strategy.

To put it on GitHub, you'll need a free account and the `git` tool, then from this folder:

```
git init
git add .
git commit -m "Options backtesting engine"
```

and follow GitHub's "create a new repository" instructions to push it. Ask me and I'll walk you through it.
