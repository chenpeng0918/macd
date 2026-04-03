import argparse
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd

try:
    import akshare as ak
except ImportError:  # pragma: no cover - data loading is optional for tests
    ak = None

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - data loading is optional for tests
    yf = None


warnings.filterwarnings("ignore")


DEFAULT_START_DATE = "2010-01-01"
DEFAULT_MARKETS = {
    "SSE Index": ("cn_index", "sh000001"),
    "Nikkei 225": ("nikkei", None),
    "Soybean Meal": ("dce", "m0"),
    "Corn": ("dce", "c0"),
    "Coke": ("dce", "j0"),
    "Iron Ore": ("dce", "i0"),
}
FUTURES_MARKETS = {"Soybean Meal", "Corn", "Coke", "Iron Ore"}


def macd_series(close, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Compute MACD indicator."""
    if isinstance(close, pd.DataFrame):
        if len(close.columns) == 1:
            close = close.iloc[:, 0]
        else:
            close = close["Close"]

    close = pd.Series(close).squeeze()

    if not hasattr(close, "index") or close.index is None:
        close = pd.Series(close.values)

    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd - macd_signal

    return pd.DataFrame(
        {
            "MACD": macd.values,
            "MACD_Signal": macd_signal.values,
            "MACD_Hist": hist.values,
        },
        index=close.index,
    )


def bbands_series(close, window: int = 20, sd: float = 2.0) -> pd.DataFrame:
    """Compute Bollinger Bands."""
    if isinstance(close, pd.DataFrame):
        if len(close.columns) == 1:
            close = close.iloc[:, 0]
        else:
            close = close["Close"]

    close = pd.Series(close).squeeze()

    if not hasattr(close, "index") or close.index is None:
        close = pd.Series(close.values)

    mid = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std(ddof=0)
    up = mid + sd * std
    lo = mid - sd * std

    return pd.DataFrame(
        {
            "BB_Middle": mid.values,
            "BB_Upper": up.values,
            "BB_Lower": lo.values,
        },
        index=close.index,
    )


def signals_macd(df: pd.DataFrame, long_short: bool = False) -> pd.Series:
    """Generate MACD crossover positions."""
    macd = df["MACD"]
    signal = df["MACD_Signal"]
    valid = macd.notna() & signal.notna()
    pos = np.where(macd > signal, 1, (-1 if long_short else 0))
    pos = np.where(np.isclose(macd, signal, equal_nan=False), 0, pos)
    pos = np.where(valid, pos, 0)
    return pd.Series(pos, index=df.index, name="position")


def signals_bbands(df: pd.DataFrame, long_short: bool = False) -> pd.Series:
    """Generate Bollinger Bands breakout positions."""
    close = df["Close"]
    up = df["BB_Upper"]
    lo = df["BB_Lower"]
    mid = df["BB_Middle"]

    pos = np.zeros(len(df), dtype=int)
    for i in range(1, len(df)):
        p = pos[i - 1]
        if long_short:
            if close.iloc[i] > up.iloc[i]:
                p = 1
            elif close.iloc[i] < lo.iloc[i]:
                p = -1
            elif (close.iloc[i - 1] >= mid.iloc[i - 1]) and (close.iloc[i] < mid.iloc[i]):
                p = 0
            elif (close.iloc[i - 1] <= mid.iloc[i - 1]) and (close.iloc[i] > mid.iloc[i]):
                p = 0
        else:
            if close.iloc[i] > up.iloc[i]:
                p = 1
            elif close.iloc[i] < mid.iloc[i]:
                p = 0
        pos[i] = p
    return pd.Series(pos, index=df.index, name="position")


def signals_combo(df: pd.DataFrame, long_short: bool = False) -> pd.Series:
    """Generate combined MACD and Bollinger Bands positions."""
    close = df["Close"]
    up = df["BB_Upper"]
    lo = df["BB_Lower"]
    mid = df["BB_Middle"]
    macd = df["MACD"]
    sig = df["MACD_Signal"]

    pos = np.zeros(len(df), dtype=int)
    for i in range(1, len(df)):
        p = pos[i - 1]
        if p == 1 and ((close.iloc[i] < mid.iloc[i]) or (macd.iloc[i] < sig.iloc[i])):
            p = 0
        if p == -1 and ((close.iloc[i] > mid.iloc[i]) or (macd.iloc[i] > sig.iloc[i])):
            p = 0
        if (close.iloc[i] > up.iloc[i]) and (macd.iloc[i] > sig.iloc[i]):
            p = 1
        elif long_short and (close.iloc[i] < lo.iloc[i]) and (macd.iloc[i] < sig.iloc[i]):
            p = -1
        pos[i] = p
    return pd.Series(pos, index=df.index, name="position")


@dataclass
class Metrics:
    cagr: float
    ann_vol: float
    sharpe: float
    maxdd: float
    win_rate: float
    n_trades: int
    total_return: float


def compute_trade_stats(position: pd.Series, market_ret: pd.Series, cost_bps: float) -> Tuple[int, float]:
    """Count completed trade segments and their win rate."""
    position = position.fillna(0.0).astype(float)
    market_ret = market_ret.fillna(0.0).astype(float)
    cost_rate = cost_bps / 10000.0
    trade_returns = []
    trade_equity = None

    for i in range(len(position)):
        curr_pos = float(position.iloc[i])
        prev_pos = float(position.iloc[i - 1]) if i > 0 else 0.0

        if prev_pos != 0.0 and curr_pos != prev_pos and trade_equity is not None:
            exit_cost = abs(prev_pos) * cost_rate
            trade_equity *= 1.0 - exit_cost
            trade_returns.append(trade_equity - 1.0)
            trade_equity = None

        if curr_pos != 0.0 and curr_pos != prev_pos:
            entry_cost = abs(curr_pos) * cost_rate
            trade_equity = 1.0
            trade_equity *= 1.0 + curr_pos * market_ret.iloc[i] - entry_cost
        elif curr_pos != 0.0 and trade_equity is not None:
            trade_equity *= 1.0 + curr_pos * market_ret.iloc[i]

    if trade_equity is not None:
        trade_returns.append(trade_equity - 1.0)

    n_trades = len(trade_returns)
    win_rate = float(np.mean(np.array(trade_returns) > 0)) if trade_returns else 0.0
    return n_trades, win_rate


def compute_metrics(
    equity: pd.Series,
    rets: pd.Series,
    position: pd.Series,
    market_ret: pd.Series,
    cost_bps: float,
) -> Metrics:
    """Compute performance metrics."""
    af = 252
    total_ret = equity.iloc[-1] / equity.iloc[0] - 1.0
    yrs = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    cagr = (1 + total_ret) ** (1 / yrs) - 1 if yrs > 0 else 0.0
    ann_vol = rets.std(ddof=0) * math.sqrt(af)
    sharpe = (rets.mean() * af) / (ann_vol + 1e-12)
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    maxdd = dd.min()
    n_trades, win_rate = compute_trade_stats(position, market_ret, cost_bps)
    return Metrics(cagr, ann_vol, sharpe, maxdd, win_rate, n_trades, total_ret)


def backtest(
    df: pd.DataFrame, cost_bps: float = 0.0, long_short: bool = False
) -> Tuple[pd.DataFrame, Metrics]:
    """Run a daily-bar backtest using next-day execution."""
    px = df["Close"].astype(float)
    signal_pos = df["position"].astype(float)
    pos = signal_pos.shift(1).fillna(0.0)
    ret = px.pct_change().fillna(0.0)
    turnover = pos.diff().abs().fillna(pos.abs())
    cost = turnover * (cost_bps / 10000.0)
    strat_ret = pos * ret - cost
    equity = (1.0 + strat_ret).cumprod()
    metrics = compute_metrics(equity, strat_ret, pos, ret, cost_bps)

    result = pd.DataFrame(
        {
            "Close": px,
            "signal_position": signal_pos,
            "position": pos,
            "ret": ret,
            "cost": cost,
            "strat_ret": strat_ret,
            "equity": equity,
            "turnover": turnover,
        },
        index=df.index,
    )
    return result, metrics


def _require_dependency(module, package_name: str) -> None:
    if module is None:
        raise ImportError(f"{package_name} is required for this operation. Install it with pip first.")


def get_matplotlib_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - plotting is optional for tests
        raise ImportError("matplotlib is required for plotting. Install it with pip first.") from exc

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def load_cn_index(symbol_code: str, start_date: str) -> pd.DataFrame:
    """Load a China index via AkShare."""
    _require_dependency(ak, "akshare")
    df_raw = ak.stock_zh_index_daily(symbol=symbol_code)
    df = df_raw.rename(
        columns={
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[df.index >= start_date]


def load_nikkei(start_date: str) -> pd.DataFrame:
    """Load Nikkei 225 via Yahoo Finance."""
    _require_dependency(yf, "yfinance")
    df = yf.download("^N225", start=start_date, auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError("Failed to load Nikkei 225 data.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    return df


def load_dce_futures(symbol: str) -> pd.DataFrame:
    """Load DCE main contract series via AkShare."""
    _require_dependency(ak, "akshare")
    df_raw = ak.futures_main_sina(symbol=symbol)
    colmap = {}
    for column in df_raw.columns:
        if "日期" in column:
            colmap[column] = "Date"
        if "开盘" in column:
            colmap[column] = "Open"
        if "最高" in column:
            colmap[column] = "High"
        if "最低" in column:
            colmap[column] = "Low"
        if "收盘" in column:
            colmap[column] = "Close"
        if "成交" in column:
            colmap[column] = "Volume"

    df = df_raw.rename(columns=colmap)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")
    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_market_data(markets: Dict[str, Tuple[str, str | None]], start_date: str) -> Dict[str, pd.DataFrame]:
    """Load all configured market data."""
    print("=" * 60)
    print("Loading market data...")
    print("=" * 60)

    data_dict = {}
    for market_name, (source, code) in markets.items():
        try:
            if source == "cn_index":
                df = load_cn_index(code, start_date)
            elif source == "nikkei":
                df = load_nikkei(start_date)
            elif source == "dce":
                df = load_dce_futures(code)
                df = df[df.index >= start_date]
            else:
                raise ValueError(f"Unknown data source: {source}")

            df = df.dropna(subset=["Close"])
            data_dict[market_name] = df
            print(f"  ✓ {market_name}: {len(df)} records")
            print(f"    Period: {df.index.min().date()} to {df.index.max().date()}")
        except Exception as exc:
            print(f"  ✗ {market_name}: Loading failed - {exc}")

    print("=" * 60)
    print(f"Data loading completed, {len(data_dict)} markets loaded successfully")
    print("=" * 60)
    return data_dict


def plot_market_analysis(df: pd.DataFrame, result: pd.DataFrame, title: str) -> None:
    """Plot market analysis with four panels."""
    plt = get_matplotlib_pyplot()
    fig, axes = plt.subplots(4, 1, figsize=(16, 12))

    ax1 = axes[0]
    ax1.plot(df.index, df["Close"], label="Close Price", linewidth=1.5, color="black")
    ax1.plot(df.index, df["BB_Upper"], "--", label="BB Upper", alpha=0.7, color="red", linewidth=1)
    ax1.plot(df.index, df["BB_Middle"], "--", label="BB Middle", alpha=0.7, color="blue", linewidth=1)
    ax1.plot(df.index, df["BB_Lower"], "--", label="BB Lower", alpha=0.7, color="red", linewidth=1)
    ax1.fill_between(df.index, df["BB_Upper"], df["BB_Lower"], alpha=0.1, color="gray")
    ax1.set_ylabel("Price", fontsize=11)
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(df.index, df["MACD"], label="MACD", linewidth=1.5, color="blue")
    ax2.plot(df.index, df["MACD_Signal"], label="Signal Line", linewidth=1.5, color="red")
    colors = ["green" if value > 0 else "red" for value in df["MACD_Hist"]]
    ax2.bar(df.index, df["MACD_Hist"], label="Histogram", alpha=0.4, color=colors, width=1)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.5)
    ax2.set_ylabel("MACD", fontsize=11)
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    ax3.plot(result.index, result["position"], linewidth=1.5, color="purple", drawstyle="steps-post")
    ax3.fill_between(result.index, 0, result["position"], alpha=0.3, color="purple", step="post")
    ax3.set_ylabel("Position", fontsize=11)
    ax3.set_ylim([-1.5, 1.5])
    ax3.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax3.grid(True, alpha=0.3)
    ax3.text(
        0.02,
        0.95,
        "1=Long, -1=Short, 0=Flat",
        transform=ax3.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    ax4 = axes[3]
    ax4.plot(result.index, result["equity"], linewidth=2, color="darkgreen", label="Strategy Equity")
    ax4.fill_between(
        result.index,
        1,
        result["equity"],
        where=(result["equity"] >= 1),
        alpha=0.3,
        color="green",
        interpolate=True,
    )
    ax4.fill_between(
        result.index,
        1,
        result["equity"],
        where=(result["equity"] < 1),
        alpha=0.3,
        color="red",
        interpolate=True,
    )
    ax4.axhline(1, color="black", linewidth=0.8, linestyle="--", alpha=0.5, label="Initial Capital")
    ax4.set_ylabel("Equity", fontsize=11)
    ax4.set_xlabel("Date", fontsize=11)
    ax4.legend(loc="best", fontsize=9)
    ax4.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.show()


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append indicators to raw OHLCV data."""
    macd = macd_series(df["Close"])
    bb = bbands_series(df["Close"])
    return pd.concat([df, macd, bb], axis=1)


def run_strategy(
    data_dict: Dict[str, pd.DataFrame],
    strategy_name: str,
    signal_fn: Callable[[pd.DataFrame, bool], pd.Series],
    cost_bps: float = 0.0,
    plot: bool = True,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Metrics]]:
    """Run one strategy across all markets."""
    print("\n" + "=" * 80)
    print(f"Strategy: {strategy_name}")
    print("=" * 80)

    results = {}
    metrics_dict = {}

    for market_name, df in data_dict.items():
        print(f"\n{'=' * 60}")
        print(f"Processing Market: {market_name}")
        print("=" * 60)

        is_futures = market_name in FUTURES_MARKETS
        feat = prepare_features(df)
        feat["position"] = signal_fn(feat, long_short=is_futures)

        result, metrics = backtest(feat, cost_bps=cost_bps, long_short=is_futures)
        results[market_name] = result
        metrics_dict[market_name] = metrics

        print("\n[Performance Metrics]")
        print(f"  Total Return:      {metrics.total_return:>8.2%}")
        print(f"  CAGR:              {metrics.cagr:>8.2%}")
        print(f"  Sharpe Ratio:      {metrics.sharpe:>8.2f}")
        print(f"  Max Drawdown:      {metrics.maxdd:>8.2%}")
        print(f"  Win Rate:          {metrics.win_rate:>8.2%}")
        print(f"  Number of Trades:  {metrics.n_trades:>8d}")

        if plot:
            plot_market_analysis(feat, result, f"{market_name} - {strategy_name}")

    print("\n" + "=" * 80)
    print(f"✓ {strategy_name} Backtest Completed")
    print("=" * 80)
    return results, metrics_dict


def build_summary_df(metrics_by_strategy: Dict[str, Dict[str, Metrics]], markets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a summary table for all strategies and markets."""
    rows = []
    for market_name in markets.keys():
        for strategy_name, metrics_dict in metrics_by_strategy.items():
            if market_name not in metrics_dict:
                continue
            metric = metrics_dict[market_name]
            rows.append(
                {
                    "Market": market_name,
                    "Strategy": strategy_name,
                    "Total Return": f"{metric.total_return:.2%}",
                    "CAGR": f"{metric.cagr:.2%}",
                    "Sharpe Ratio": f"{metric.sharpe:.2f}",
                    "Max Drawdown": f"{metric.maxdd:.2%}",
                    "Win Rate": f"{metric.win_rate:.2%}",
                    "Trades": metric.n_trades,
                }
            )
    return pd.DataFrame(rows)


def build_descriptive_stats_df(
    results_by_strategy: Dict[str, Dict[str, pd.DataFrame]], markets: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Build descriptive stats for strategy returns."""
    rows = []
    for market_name in markets.keys():
        for strategy_name, results_dict in results_by_strategy.items():
            if market_name not in results_dict:
                continue
            returns = results_dict[market_name]["strat_ret"].dropna()
            rows.append(
                {
                    "Market": market_name,
                    "Strategy": strategy_name,
                    "Mean": returns.mean(),
                    "Std Dev": returns.std(),
                    "Min": returns.min(),
                    "Max": returns.max(),
                    "Median": returns.median(),
                    "Skewness": returns.skew(),
                    "Kurtosis": returns.kurtosis(),
                }
            )
    return pd.DataFrame(rows)


def save_reports(
    summary_df: pd.DataFrame, descriptive_df: pd.DataFrame, output_dir: Path
) -> Tuple[Path, Path]:
    """Save CSV reports into the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "strategy_performance_summary.csv"
    descriptive_path = output_dir / "return_descriptive_statistics.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    descriptive_df.to_csv(descriptive_path, index=False, encoding="utf-8-sig")
    return summary_path, descriptive_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest MACD and Bollinger Band strategies.")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--cost-bps", type=float, default=0.0, help="One-way transaction cost in basis points.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for exported CSV reports.")
    parser.add_argument("--no-plots", action="store_true", help="Disable matplotlib charts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dict = load_market_data(DEFAULT_MARKETS, args.start_date)

    strategies = {
        "MACD": signals_macd,
        "Bollinger Bands": signals_bbands,
        "Combined": signals_combo,
    }

    results_by_strategy = {}
    metrics_by_strategy = {}

    for strategy_name, signal_fn in strategies.items():
        results, metrics_dict = run_strategy(
            data_dict,
            strategy_name=strategy_name,
            signal_fn=signal_fn,
            cost_bps=args.cost_bps,
            plot=not args.no_plots,
        )
        results_by_strategy[strategy_name] = results
        metrics_by_strategy[strategy_name] = metrics_dict

    summary_df = build_summary_df(metrics_by_strategy, data_dict)
    descriptive_df = build_descriptive_stats_df(results_by_strategy, data_dict)
    summary_path, descriptive_path = save_reports(summary_df, descriptive_df, Path(args.output_dir))

    print("\n✓ Performance summary saved to:", summary_path)
    print("✓ Descriptive statistics saved to:", descriptive_path)


if __name__ == "__main__":
    main()
