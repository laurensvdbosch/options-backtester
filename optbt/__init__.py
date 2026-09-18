"""optbt: a modular options-strategy backtesting engine.

Layers (each depends only on the ones above it):

    pricing     -> Black-Scholes price, Greeks and implied-vol inversion
    instruments -> contracts, quotes, positions
    market      -> simulate the underlying, build an implied-vol surface, build option chains
    execution   -> spread / slippage / commission models and the simulated broker
    strategy    -> the Strategy interface and example strategies
    engine      -> the daily event loop tying it all together
    analytics   -> performance metrics and Greek-exposure series
    reporting   -> text / markdown summaries and charts
"""

from optbt.config import BacktestConfig, ExecutionConfig, MarketConfig
from optbt.engine import Backtester, BacktestResult

__all__ = [
    "Backtester",
    "BacktestResult",
    "BacktestConfig",
    "MarketConfig",
    "ExecutionConfig",
]

__version__ = "0.1.0"
