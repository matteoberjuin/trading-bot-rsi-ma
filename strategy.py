

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')


CONFIG = {
    "ticker":        "SPY",
    "start": "2023-01-01",
    "end":   "2024-12-31",
    "initial_cash":  10_000,

    "rsi_period":    14,
    "rsi_oversold":  40,
    "rsi_overbought": 65,

    "ma_fast":       20,
    "ma_slow":       50,

    "position_size": 1.0,
}



def compute_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs  = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def add_indicators(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()
    df["MA_fast"] = df["Close"].rolling(cfg["ma_fast"]).mean()
    df["MA_slow"] = df["Close"].rolling(cfg["ma_slow"]).mean()
    df["RSI"]     = compute_rsi(df["Close"], cfg["rsi_period"])

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"]   = df["MACD"] - df["MACD_signal"]

    return df



#  BUY  : (MACD_hist croise 0 par le haut ET prix > MA_slow) OU (RSI remonte au dessus de 40 ET MACD_hist positif)
#  SELL : (MACD_hist croise 0 par le bas) OU (RSI > 65) OU (prix passe sous MA_slow)



def generate_signals(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:

    df = df.copy()
    df["signal"] = 0


    cond_ma = df["Close"] > df["MA_slow"]

    cond_rsi = (df["RSI"] > 40) & (df["RSI"].shift(1) <= 40)

    cond_macd = (df["MACD_hist"] > 0) & (df["MACD_hist"].shift(1) <= 0)

    buy_signal = cond_macd & cond_ma
    buy_signal2 = cond_rsi & (df["MACD_hist"] > 0)

    df.loc[buy_signal | buy_signal2, "signal"] = 1

    sell_macd = (df["MACD_hist"] < 0) & (df["MACD_hist"].shift(1) >= 0)

    sell_rsi  = df["RSI"] > cfg["rsi_overbought"]

    sell_ma   = (df["Close"] < df["MA_slow"]) & (df["Close"].shift(1) >= df["MA_slow"].shift(1))

    df.loc[sell_macd | sell_rsi | sell_ma, "signal"] = -1

    return df




def run_backtest(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    cash       = cfg["initial_cash"]
    shares     = 0
    position   = 0          # 0 = out, 1 = in
    trades     = []
    portfolio  = []
    entry_price = 0

    for date, row in df.iterrows():
        price = row["Close"]
        sig   = row["signal"]

        if sig == 1 and position == 0 and not np.isnan(row["MA_slow"]):
            shares      = (cash * cfg["position_size"]) / price
            cash       -= shares * price
            position    = 1
            entry_price = price
            trades.append({"date": date, "type": "BUY", "price": price, "shares": shares})

        elif sig == -1 and position == 1:
            cash    += shares * price
            pnl      = (price - entry_price) * shares
            trades.append({"date": date, "type": "SELL", "price": price,
                           "shares": shares, "pnl": pnl,
                           "return_%": (price / entry_price - 1) * 100})
            shares   = 0
            position = 0

        portfolio_value = cash + shares * price
        portfolio.append({"date": date, "value": portfolio_value, "price": price, "position": position})


    if position == 1:
        price = df["Close"].iloc[-1]
        cash += shares * price
        pnl   = (price - entry_price) * shares
        trades.append({"date": df.index[-1], "type": "SELL", "price": price,
                       "shares": shares, "pnl": pnl,
                       "return_%": (price / entry_price - 1) * 100})

    portfolio_df = pd.DataFrame(portfolio).set_index("date")
    trades_df    = pd.DataFrame(trades)
    return portfolio_df, trades_df



def compute_metrics(portfolio_df: pd.DataFrame, trades_df: pd.DataFrame, cfg: dict) -> dict:
    values = portfolio_df["value"]
    initial = cfg["initial_cash"]
    final   = values.iloc[-1]


    total_return = (final / initial - 1) * 100

  
    n_years = (values.index[-1] - values.index[0]).days / 365.25
    cagr    = ((final / initial) ** (1 / n_years) - 1) * 100

    
    daily_ret = (values - values.shift(1)) / values.shift(1)
    daily_ret = daily_ret.dropna()

    risk_free_daily = 0.05 / 252  
    sharpe = ((daily_ret.mean() - risk_free_daily) / daily_ret.std()) * np.sqrt(252) if daily_ret.std() > 0 else 0


    roll_max  = values.cummax()
    drawdown  = (values - roll_max) / roll_max
    max_dd    = drawdown.min() * 100

  
    bh_return = (portfolio_df["price"].iloc[-1] / portfolio_df["price"].iloc[0] - 1) * 100


    sell_trades = trades_df[trades_df["type"] == "SELL"]
    n_trades    = len(sell_trades)
    win_rate    = (sell_trades["pnl"] > 0).mean() * 100 if n_trades > 0 else 0
    avg_return  = sell_trades["return_%"].mean() if n_trades > 0 else 0

    return {
        "Total Return (%)":     round(total_return, 2),
        "CAGR (%)":             round(cagr, 2),
        "Sharpe Ratio":         round(sharpe, 3),
        "Max Drawdown (%)":     round(max_dd, 2),
        "Buy & Hold Return (%)":round(bh_return, 2),
        "Number of Trades":     n_trades,
        "Win Rate (%)":         round(win_rate, 2),
        "Avg Trade Return (%)": round(avg_return, 2),
        "Final Portfolio ($)":  round(final, 2),
    }


# ─────────────────────────────────────────
#  VISUALISATION
# ─────────────────────────────────────────
def plot_results(df: pd.DataFrame, portfolio_df: pd.DataFrame,
                 trades_df: pd.DataFrame, metrics: dict, cfg: dict):

    fig = plt.figure(figsize=(16, 12), facecolor="#0f0f1a")
    fig.suptitle(f"  Backtest — {cfg['ticker']}  |  RSI({cfg['rsi_period']}) + MM{cfg['ma_fast']}/MM{cfg['ma_slow']}",
                 fontsize=15, color="white", fontweight="bold", x=0.02, ha="left")

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3,
                           top=0.93, bottom=0.06, left=0.06, right=0.97)

    ACCENT   = "#00d4ff"
    GREEN    = "#00e676"
    RED      = "#ff5252"
    YELLOW   = "#ffd740"
    BG_PANEL = "#161627"
    GRID_C   = "#2a2a3e"

    ax_price = fig.add_subplot(gs[0, :])
    ax_rsi   = fig.add_subplot(gs[1, 0])
    ax_port  = fig.add_subplot(gs[1, 1])
    ax_dist  = fig.add_subplot(gs[2, 0])
    ax_info  = fig.add_subplot(gs[2, 1])

    for ax in [ax_price, ax_rsi, ax_port, ax_dist, ax_info]:
        ax.set_facecolor(BG_PANEL)
        ax.tick_params(colors="gray", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color(GRID_C)

    # — Price + MAs ————————————————————
    ax_price.plot(df.index, df["Close"],   color="white",  lw=0.8, alpha=0.7, label="Price")
    ax_price.plot(df.index, df["MA_fast"], color=YELLOW,   lw=1.2, label=f"MM{cfg['ma_fast']}")
    ax_price.plot(df.index, df["MA_slow"], color=ACCENT,   lw=1.2, label=f"MM{cfg['ma_slow']}")

    buys  = trades_df[trades_df["type"] == "BUY"]
    sells = trades_df[trades_df["type"] == "SELL"]
    ax_price.scatter(buys["date"],  df.loc[buys["date"],  "Close"], marker="^", color=GREEN, s=60, zorder=5, label="Buy")
    ax_price.scatter(sells["date"], df.loc[sells["date"], "Close"], marker="v", color=RED,   s=60, zorder=5, label="Sell")

    ax_price.set_title("Price & Signals", color="white", fontsize=10, pad=6)
    ax_price.legend(fontsize=7, facecolor=BG_PANEL, labelcolor="white", loc="upper left", ncol=5)
    ax_price.yaxis.grid(True, color=GRID_C, lw=0.5)

    # — RSI ———————————————————————————
    ax_rsi.plot(df.index, df["RSI"], color=ACCENT, lw=1)
    ax_rsi.axhline(cfg["rsi_overbought"], color=RED,   lw=0.8, ls="--", alpha=0.7)
    ax_rsi.axhline(cfg["rsi_oversold"],   color=GREEN, lw=0.8, ls="--", alpha=0.7)
    ax_rsi.fill_between(df.index, df["RSI"], cfg["rsi_overbought"],
                        where=(df["RSI"] > cfg["rsi_overbought"]),
                        color=RED, alpha=0.15)
    ax_rsi.fill_between(df.index, df["RSI"], cfg["rsi_oversold"],
                        where=(df["RSI"] < cfg["rsi_oversold"]),
                        color=GREEN, alpha=0.15)
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_title("RSI", color="white", fontsize=10, pad=6)
    ax_rsi.yaxis.grid(True, color=GRID_C, lw=0.5)

    # — Portfolio vs Buy & Hold ———————
    port_norm = portfolio_df["value"] / cfg["initial_cash"] * 100
    bh_norm   = portfolio_df["price"] / portfolio_df["price"].iloc[0] * 100
    ax_port.plot(portfolio_df.index, port_norm, color=GREEN, lw=1.5, label="Strategy")
    ax_port.plot(portfolio_df.index, bh_norm,   color="gray", lw=1.0, ls="--", label="Buy & Hold")
    ax_port.set_title("Strategy vs Buy & Hold (indexed 100)", color="white", fontsize=10, pad=6)
    ax_port.legend(fontsize=7, facecolor=BG_PANEL, labelcolor="white")
    ax_port.yaxis.grid(True, color=GRID_C, lw=0.5)

    # — Trade return distribution ———
    sell_returns = trades_df[trades_df["type"] == "SELL"]["return_%"].dropna()
    if len(sell_returns) > 0:
        colors = [GREEN if r > 0 else RED for r in sell_returns]
        ax_dist.bar(range(len(sell_returns)), sell_returns.values, color=colors, width=0.7)
        ax_dist.axhline(0, color="white", lw=0.6)
        ax_dist.set_title("Return per Trade (%)", color="white", fontsize=10, pad=6)
        ax_dist.set_xlabel("Trade #", color="gray", fontsize=8)
        ax_dist.yaxis.grid(True, color=GRID_C, lw=0.5)

    # — Metrics table ——————————————
    ax_info.axis("off")
    ax_info.set_title("Performance Summary", color="white", fontsize=10, pad=6)
    y = 0.95
    for k, v in metrics.items():
        color = "white"
        if "Return" in k or "CAGR" in k:
            color = GREEN if str(v).replace("-","").replace(".","").isdigit() and float(v) > 0 else RED
        if "Drawdown" in k:
            color = RED
        if "Sharpe" in k:
            color = GREEN if float(v) > 1 else YELLOW
        ax_info.text(0.02, y, k,     color="gray",  fontsize=8.5, transform=ax_info.transAxes)
        ax_info.text(0.72, y, str(v), color=color, fontsize=8.5, transform=ax_info.transAxes, fontweight="bold")
        y -= 0.095

    plt.savefig("outputs/backtest_result.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    print("✅ Chart saved → backtest_result.png")
    plt.close()


# ─────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────
def main():
    cfg = CONFIG
    print(f"\n📥 Downloading {cfg['ticker']} data ({cfg['start']} → {cfg['end']})...")
    raw = yf.download(cfg["ticker"], start=cfg["start"], end=cfg["end"], auto_adjust=True, progress=False)

    if raw.empty:
        print("Aucune donnée récupérée")
        return


    # Flatten MultiIndex columns if present
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = add_indicators(raw, cfg)
    df = generate_signals(df, cfg)

    print("🔄 Running backtest...")
    portfolio_df, trades_df = run_backtest(df, cfg)

    metrics = compute_metrics(portfolio_df, trades_df, cfg)

    print("\n" + "─" * 42)
    print(f"  BACKTEST RESULTS — {cfg['ticker']}")
    print("─" * 42)
    for k, v in metrics.items():
        print(f"  {k:<28} {v}")
    print("─" * 42)

    print("\n📊 Generating chart...")
    plot_results(df, portfolio_df, trades_df, metrics, cfg)

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
