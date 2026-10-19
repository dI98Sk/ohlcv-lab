"""Final report: the figures and the table of the README, over a fixed period.

The raw layer keeps growing (every fetch adds the newest candles), so "all the
data" means something different every week, and so would every number built
on it. The report pins its period instead: ``[START, END)``, three full years
of hourly candles. A fresh clone that runs ``make data`` gets the same table.

Figures are built with the object API (``Figure``, no ``pyplot``): no global
state, no GUI backend needed, so the same code runs in the notebook, the CLI
and the tests.
"""

from __future__ import annotations

import argparse
import functools
import logging
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from scipy import stats as st

from ohlcv_lab.bybit import interval_ms
from ohlcv_lab.stats import TAIL_SIGMA, drawdown, log_returns, rolling_vol, summary
from ohlcv_lab.storage import DEFAULT_CLEAN_DIR, load_clean

log = logging.getLogger(__name__)

START = pd.Timestamp("2023-10-01", tz="UTC")
END = pd.Timestamp("2026-10-01", tz="UTC")
INTERVAL = "60"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"]

# Colour follows the coin, in this fixed order, never its rank in a chart.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, MUTED = "#0b0b0b", "#52514e"
STYLE = {
    "figure.figsize": (11, 4.5),
    "figure.dpi": 110,
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.titlelocation": "left",
    "grid.color": "#e1e0d9",
    "grid.linewidth": 0.6,
    "xtick.color": "#898781",
    "ytick.color": "#898781",
    "text.color": INK,
    "lines.linewidth": 1.6,
    "legend.frameon": False,
}
PERCENT = FuncFormatter(lambda y, _: f"{y:.0%}")


def styled(plot):
    """Build the figure inside the report style, without touching global rcParams."""

    @functools.wraps(plot)
    def wrapper(*args, **kwargs) -> Figure:
        with mpl.rc_context(STYLE):
            return plot(*args, **kwargs)

    return wrapper


def colors(symbols: list[str]) -> dict[str, str]:
    return dict(zip(symbols, PALETTE, strict=False))


def label(symbol: str) -> str:
    return symbol.removesuffix("USDT")


def load_period(
    symbols: list[str] = SYMBOLS,
    interval: str = INTERVAL,
    start: pd.Timestamp = START,
    end: pd.Timestamp = END,
    clean_dir: Path = DEFAULT_CLEAN_DIR,
) -> pd.DataFrame:
    """Closes on ``[start, end)``; fails if the clean layer does not cover the whole period.

    A silently shorter period would give different numbers that look just as right.
    Checked per coin, before the join: after an outer join one coin's full grid
    would hide another coin listed later (its hours would just be NaN).
    """
    step = pd.Timedelta(milliseconds=interval_ms(interval))
    expected = pd.date_range(start, end, freq=step, inclusive="left")
    columns = {}
    for symbol in symbols:
        close = load_clean(symbol, interval, clean_dir)["close"]
        close = close[(close.index >= start) & (close.index < end)]
        # Compare values: Index.equals is False for the same instants in ms vs us units.
        if len(close) != len(expected) or not (close.index == expected).all():
            have = f"{close.index.min()} .. {close.index.max()}" if len(close) else "nothing"
            raise ValueError(
                f"{symbol}: clean layer has {have} for [{start}, {end}); "
                "fetch the period first (make data)"
            )
        columns[symbol] = close
    return pd.DataFrame(columns)


def summary_table(closes: pd.DataFrame, interval: str = INTERVAL) -> pd.DataFrame:
    """The README table: one row per coin, numbers unformatted."""
    table = summary(closes, interval)
    cols = ["ann_vol", "excess_kurtosis", "var_1%", "var_1%_normal", "max_drawdown", "tail_share"]
    table = table[cols].astype(float)
    table.index = [label(s) for s in table.index]
    return table


def markdown_table(table: pd.DataFrame) -> str:
    """Format ``summary_table`` as a Markdown table for the README."""
    lines = [
        "| монета | год. волатильность | эксцесс | VaR 1% | VaR 1% (нормальный) | макс. просадка |",
        "|--------|------:|-----:|-------:|-------:|-----:|",
    ]
    for coin, r in table.iterrows():
        cells = [
            f"{r['ann_vol']:.0%}",
            f"{r['excess_kurtosis']:.1f}",
            f"{r['var_1%']:.2%}",
            f"{r['var_1%_normal']:.2%}",
            f"{r['max_drawdown']:.0%}",
        ]
        cells = [c.replace("-", "−") for c in cells]  # typographic minus, as in the README
        lines.append(f"| {coin} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


@styled
def plot_growth(closes: pd.DataFrame) -> Figure:
    """What $1 bought at the start is worth over time; log scale, so equal moves look equal."""
    fig = Figure()
    ax = fig.subplots()
    for symbol, color in colors(list(closes)).items():
        close = closes[symbol]
        ax.plot(close.index, close / close.dropna().iloc[0], color=color, label=label(symbol))
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"${y:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_title("Во что превратился $1 (лог. шкала)")
    ax.legend(ncol=len(closes.columns), loc="upper left")
    return fig


@styled
def plot_vol(closes: pd.DataFrame, interval: str = INTERVAL) -> Figure:
    fig = Figure()
    ax = fig.subplots()
    for symbol, color in colors(list(closes)).items():
        vol = rolling_vol(log_returns(closes[symbol]), interval, "30D")
        ax.plot(vol.index, vol, color=color, label=label(symbol))
    ax.yaxis.set_major_formatter(PERCENT)
    ax.set_title("Скользящая годовая волатильность, окно 30 дней")
    ax.legend(ncol=len(closes.columns), loc="upper left")
    return fig


@styled
def plot_drawdowns(closes: pd.DataFrame) -> Figure:
    """One panel per coin on a shared axis, so depths compare at a glance."""
    n = len(closes.columns)
    fig = Figure(figsize=(11, 1.6 * n + 0.4))
    axes = np.atleast_1d(fig.subplots(n, 1, sharex=True, sharey=True))
    for ax, (symbol, color) in zip(axes, colors(list(closes)).items(), strict=False):
        dd = drawdown(closes[symbol])
        ax.fill_between(dd.index, dd, 0, color=color, alpha=0.35, linewidth=0)
        ax.plot(dd.index, dd, color=color, linewidth=1)
        ax.annotate(
            f"{dd.min():.0%}",
            (dd.idxmin(), dd.min()),
            xytext=(0, -11),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=MUTED,
        )
        ax.set_ylim(-1, 0.05)
        ax.set_ylabel(label(symbol), rotation=0, ha="right", va="center")
        ax.yaxis.set_major_formatter(PERCENT)
    axes[0].set_title("Просадка от исторического максимума")
    return fig


@styled
def plot_tails(closes: pd.DataFrame, symbol: str | None = None) -> Figure:
    """Standardised returns against the normal density; log y, or the tails are invisible."""
    symbol = symbol or closes.columns[0]
    r = log_returns(closes[symbol]).dropna()
    z = (r - r.mean()) / r.std()
    expected = len(z) * 2 * st.norm.sf(TAIL_SIGMA)
    beyond = int((z.abs() > TAIL_SIGMA).sum())

    fig = Figure()
    ax = fig.subplots()
    edge = 12
    ax.hist(
        z.clip(-edge, edge),
        bins=np.linspace(-edge, edge, 121),
        density=True,
        color=colors(list(closes))[symbol],
        alpha=0.55,
        label=f"{label(symbol)}, факт",
    )
    x = np.linspace(-edge, edge, 500)
    ax.plot(x, st.norm.pdf(x), color=INK, linewidth=1.2, linestyle="--", label="нормальное")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1)
    ax.set_xlabel("доходность за свечу, в σ")
    ax.set_title(
        f"{label(symbol)}: движений > {TAIL_SIGMA:g}σ {beyond} из {len(z)}, "
        f"нормальная модель ждёт {expected:.1f}"
    )
    ax.legend()
    return fig


@styled
def plot_corr(closes: pd.DataFrame) -> Figure:
    corr = closes.apply(log_returns).corr()
    labels = [label(s) for s in corr.columns]
    fig = Figure(figsize=(5.5, 4.5))
    ax = fig.subplots()
    im = ax.imshow(corr, cmap="Blues", vmin=0.5, vmax=1)  # from 0.5, or nothing differs
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.grid(False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = corr.iloc[i, j]
            color = "white" if v > 0.8 else INK
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9, color=color)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Корреляция доходностей")
    return fig


FIGURES = {
    "growth": plot_growth,
    "volatility": plot_vol,
    "drawdowns": plot_drawdowns,
    "tails": plot_tails,
    "correlation": plot_corr,
}


def build(closes: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Save every figure as PNG into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, plot in FIGURES.items():
        path = out_dir / f"{name}.png"
        plot(closes).savefig(path, bbox_inches="tight")
        paths.append(path)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ohlcv-report",
        description="README figures and table over the fixed report period.",
    )
    parser.add_argument("--clean-dir", type=Path, default=DEFAULT_CLEAN_DIR)
    parser.add_argument("--out", type=Path, default=Path("docs/img"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")

    closes = load_period(clean_dir=args.clean_dir)
    for path in build(closes, args.out):
        log.info("wrote %s", path)
    print(f"{INTERVAL} candles, [{START:%Y-%m-%d}, {END:%Y-%m-%d}), {len(closes)} per coin\n")
    print(markdown_table(summary_table(closes)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
