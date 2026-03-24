import ccxt
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
API_KEY = "OFyvnhMAfNb14JsVb1JgsIUYPzTNkNLm4xLpIxzhYwYo3PsrhbeBsapKyfvqxzvn"
API_SECRET = "rR9R392Eb5rNj0WOEimnjy1aU2O6D4qS05dUBwmhwCsRHgcpk6kdq9rLdZWmMoAi"
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1m' 
TRADE_SIZE = 0.001 
LOG_FILE = "trades_log.csv"

CONFIG = {
    "rsi_period": 25,
    "rsi_oversold": 48,
    "rsi_overbought": 80,
    "ma_fast": 20,
    "ma_slow": 50,
    "atr_period": 14,      
    "atr_multiplier": 3.0,
}

# --- INITIALISATION ---
exchange = ccxt.binance({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
    'options': {'defaultType': 'spot'}
})
exchange.set_sandbox_mode(True) 

def log_trade(type, price, reason):
    """Enregistre le trade dans un fichier CSV"""
    df_log = pd.DataFrame([{
        "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "type": type,
        "price": price,
        "reason": reason
    }])
    # Si le fichier n'existe pas, on met l'entête, sinon on ajoute à la suite
    df_log.to_csv(LOG_FILE, mode='a', header=not os.path.exists(LOG_FILE), index=False)

def compute_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def add_indicators(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()
    df["MA_slow"] = df["Close"].rolling(cfg["ma_slow"]).mean()
    df["RSI"] = compute_rsi(df["Close"], cfg["rsi_period"])
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD_hist"] = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()
    df["TR"] = pd.concat([df["High"]-df["Low"], (df["High"]-df["Close"].shift(1)).abs(), (df["Low"]-df["Close"].shift(1)).abs()], axis=1).max(axis=1)
    df["ATR"] = df["TR"].rolling(cfg["atr_period"]).mean()
    return df

def main():
    print(f"🚀 Bot en ligne sur {SYMBOL} ({TIMEFRAME})")
    in_position = False 
    trailing_stop = 0.0

    while True:
        try:
            bars = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=100)
            df = pd.DataFrame(bars, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df = add_indicators(df, CONFIG)
            
            last = df.iloc[-2]
            prev = df.iloc[-3]
            current_price = df.iloc[-1]['Close']
            current_atr = last['ATR']

            print(f"\r[{datetime.now().strftime('%H:%M:%S')}] Prix: {current_price:.2f} | RSI: {last['RSI']:.2f}", end="")

            if in_position:
                new_stop = current_price - (CONFIG["atr_multiplier"] * current_atr)
                trailing_stop = max(trailing_stop, new_stop)

                if current_price <= trailing_stop:
                    print(f"\n STOP-LOSS TOUCHÉ ! Vente à {current_price}")
                    exchange.create_market_sell_order(SYMBOL, TRADE_SIZE)
                    log_trade("SELL", current_price, "Trailing Stop")
                    in_position = False
                    trailing_stop = 0.0
                    continue

            cond_ma = last["Close"] > last["MA_slow"]
            cond_rsi_cross = (last["RSI"] > CONFIG["rsi_oversold"]) and (prev["RSI"] <= CONFIG["rsi_oversold"])
            cond_macd_cross = (last["MACD_hist"] > 0) and (prev["MACD_hist"] <= 0)

            if not in_position and ((cond_macd_cross and cond_ma) or (cond_rsi_cross and last["MACD_hist"] > 0 and cond_ma)):
                print(f"\n🟢 ACHAT : Ordre envoyé à {current_price}")
                exchange.create_market_buy_order(SYMBOL, TRADE_SIZE)
                log_trade("BUY", current_price, "Signal Strategie")
                in_position = True
                trailing_stop = current_price - (CONFIG["atr_multiplier"] * current_atr)

            elif in_position and ((last["MACD_hist"] < 0 and prev["MACD_hist"] >= 0) or (last["RSI"] > CONFIG["rsi_overbought"])):
                print(f"\n🔴 VENTE : Signal de sortie à {current_price}")
                exchange.create_market_sell_order(SYMBOL, TRADE_SIZE)
                log_trade("SELL", current_price, "Signal Sortie")
                in_position = False
                trailing_stop = 0.0

            time.sleep(10) 

        except Exception as e:
            print(f"\n❌ Erreur : {e}")
            time.sleep(30)

if __name__ == "__main__":
    main()