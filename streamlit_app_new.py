import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import plotly.graph_objects as go
import xml.etree.ElementTree as ET
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(page_title="AI Trading Signal App v4", layout="wide", page_icon="📈")

st.title("📈 AI Trading Signal App v4")
st.caption("Market Direction · News Sentiment · Support/Resistance · Multi-Timeframe · Scanner Buckets")
st.warning("⚠️ Educational only. Not financial advice. Signals are not guaranteed.")

DEFAULT_TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","GOOG","META","TSLA","NFLX","AMD","SPY","QQQ",
    "PLTR","SMCI","AVGO","CRM","MU","INTC","UBER","SHOP","COIN","SOFI","PYPL","ADBE",
    "PANW","MSTR","ARM","BABA","DIS","NKE","COST","WMT","TGT","JPM","BAC","V","MA",
    "UNH","LLY","XOM","CVX","HOOD","RIVN","LCID","NIO","F","GM","BA","GE","T","VZ",
    "PFE","MRNA","KO","PEP","SBUX","ORCL","IBM"
]

POS_WORDS = ["beat","beats","upgrade","upgraded","bullish","surge","surges","rally","record",
             "strong","growth","profit","ai","partnership","raises","buy","outperform","positive"]
NEG_WORDS = ["miss","misses","downgrade","downgraded","bearish","falls","drop","plunge","weak",
             "loss","lawsuit","probe","cuts","cut","sell","underperform","negative","warning","concern"]

def safe_num(x, default=0.0):
    try:
        if x is None:
            return default
        if isinstance(x, float) and np.isnan(x):
            return default
        if hasattr(x, "__iter__") and not isinstance(x, str):
            return default
        return float(x)
    except Exception:
        return default

@st.cache_data(ttl=86400)
def get_all_tickers():
    tickers = set(DEFAULT_TICKERS)
    urls = [
        "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=10)
            for line in r.text.splitlines()[1:]:
                parts = line.split("|")
                if len(parts) > 1:
                    sym = parts[0].strip()
                    if sym.isalpha() and 1 <= len(sym) <= 5:
                        tickers.add(sym)
        except Exception:
            pass
    return sorted(tickers)

ALL_TICKERS = get_all_tickers()

def _clean(df, ticker):
    if df is None or df.empty:
        return pd.DataFrame(), f"No data for {ticker}"
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.reset_index()
    for alias in ("Datetime", "index"):
        if "Date" not in df.columns and alias in df.columns:
            df = df.rename(columns={alias: "Date"})
    if "Date" not in df.columns:
        df.insert(0, "Date", pd.to_datetime(df.index))
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]
    needed = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        return pd.DataFrame(), f"Missing columns: {missing}"
    df = df[needed].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("Date").dropna(subset=["Date", "Close"])
    return (df, "") if not df.empty else (pd.DataFrame(), "Empty after cleaning")

@st.cache_data(ttl=1800)
def load_price_data(ticker, period="6mo", interval="1d"):
    try:
        raw = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
            group_by="column"
        )
        df, err = _clean(raw, ticker)
        if not df.empty:
            return df, ""
        return pd.DataFrame(), f"No data returned for {ticker}. Yahoo may be rate-limiting."
    except Exception as e:
        msg = str(e)
        if "Too Many Requests" in msg or "Rate limited" in msg or "429" in msg:
            return pd.DataFrame(), "Yahoo Finance rate limit reached. Wait 10-30 minutes and try again."
        return pd.DataFrame(), msg

def add_indicators(df):
    df = df.copy()
    c = df["Close"]
    h = df["High"]
    lo = df["Low"]
    v = df["Volume"]

    df["Return"] = c.pct_change()
    df["MA9"] = c.rolling(9, min_periods=3).mean()
    df["MA20"] = c.rolling(20, min_periods=5).mean()
    df["MA50"] = c.rolling(50, min_periods=10).mean()
    df["MA200"] = c.rolling(120, min_periods=20).mean()
    df["EMA9"] = c.ewm(span=9, adjust=False).mean()
    df["EMA21"] = c.ewm(span=21, adjust=False).mean()

    delta = c.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14, min_periods=5).mean()
    avg_loss = loss.rolling(14, min_periods=5).mean().replace(0, np.nan)
    df["RSI"] = 100 - 100 / (1 + avg_gain / avg_loss)

    rsi_min = df["RSI"].rolling(14, min_periods=5).min()
    rsi_max = df["RSI"].rolling(14, min_periods=5).max()
    stoch_k = 100 * (df["RSI"] - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)
    df["STOCH_K"] = stoch_k.rolling(3, min_periods=1).mean()
    df["STOCH_D"] = df["STOCH_K"].rolling(3, min_periods=1).mean()

    exp12 = c.ewm(span=12, adjust=False).mean()
    exp26 = c.ewm(span=26, adjust=False).mean()
    df["MACD"] = exp12 - exp26
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]

    tr1 = h - lo
    tr2 = (h - c.shift()).abs()
    tr3 = (lo - c.shift()).abs()
    df["TR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR"] = df["TR"].rolling(14, min_periods=5).mean()

    bb_mid = c.rolling(20, min_periods=5).mean()
    bb_std = c.rolling(20, min_periods=5).std()
    df["BB_MID"] = bb_mid
    df["BB_UPPER"] = bb_mid + 2 * bb_std
    df["BB_LOWER"] = bb_mid - 2 * bb_std
    df["BB_PCT"] = (c - df["BB_LOWER"]) / (df["BB_UPPER"] - df["BB_LOWER"]).replace(0, np.nan)

    df["Volume_MA20"] = v.rolling(20, min_periods=5).mean()
    df["Volume_Ratio"] = v / df["Volume_MA20"].replace(0, np.nan)

    obv = [0]
    closes = c.values
    vols = v.values
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            obv.append(obv[-1] + vols[i])
        elif closes[i] < closes[i - 1]:
            obv.append(obv[-1] - vols[i])
        else:
            obv.append(obv[-1])
    df["OBV"] = obv
    df["OBV_MA20"] = pd.Series(obv, index=df.index).rolling(20, min_periods=5).mean()
    df["OBV_Trend"] = df["OBV"] - df["OBV_MA20"]

    df["Support"] = c.rolling(30, min_periods=5).min()
    df["Resistance"] = c.rolling(30, min_periods=5).max()
    df["Volatility"] = df["Return"].rolling(20, min_periods=5).std() * np.sqrt(252)

    pdm = h.diff()
    mdm = -lo.diff()
    pdm = pdm.where((pdm > mdm) & (pdm > 0), 0)
    mdm = mdm.where((mdm > pdm) & (mdm > 0), 0)
    atr14 = df["TR"].rolling(14, min_periods=5).mean()
    df["PLUS_DI"] = 100 * pdm.rolling(14, min_periods=5).mean() / atr14.replace(0, np.nan)
    df["MINUS_DI"] = 100 * mdm.rolling(14, min_periods=5).mean() / atr14.replace(0, np.nan)
    dx = 100 * (df["PLUS_DI"] - df["MINUS_DI"]).abs() / (df["PLUS_DI"] + df["MINUS_DI"]).replace(0, np.nan)
    df["ADX"] = dx.rolling(14, min_periods=5).mean()

    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    return df.dropna()

def resample_weekly(df):
    w = df.set_index("Date").resample("W-FRI").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
    }).dropna().reset_index()
    return w

def detect_buysell_signals(df):
    signals = []
    df = df.reset_index(drop=True)
    prev_ema_diff = df["EMA9"] - df["EMA21"]

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]
        ema_diff = row["EMA9"] - row["EMA21"]
        prev_diff = prev_ema_diff.iloc[i - 1]

        ema_cross_up = (prev_diff <= 0) and (ema_diff > 0)
        rsi_ok = row["RSI"] < 65
        macd_bullish = row["MACD_HIST"] > 0
        obv_bullish = row["OBV_Trend"] > 0

        if ema_cross_up and rsi_ok and macd_bullish and obv_bullish:
            signals.append({"Date": row["Date"], "Price": row["Close"], "Type": "BUY",
                            "Reason": "EMA9 crossed above EMA21 + MACD bullish + OBV inflow"})
            continue

        ema_cross_dn = (prev_diff >= 0) and (ema_diff < 0)
        double_ob = (row["RSI"] > 70) and (row["STOCH_K"] > 80)
        macd_bearish = (row["MACD_HIST"] < 0) and (prev_row["MACD_HIST"] >= 0)

        if ema_cross_dn or double_ob or macd_bearish:
            signals.append({"Date": row["Date"], "Price": row["Close"], "Type": "SELL",
                            "Reason": "EMA cross down" if ema_cross_dn else
                                      "RSI+Stoch double overbought" if double_ob else
                                      "MACD histogram turned negative"})
    return signals

@st.cache_data(ttl=1800)
def get_market_direction():
    rows = []
    scores = []
    for ticker in ["SPY", "QQQ"]:
        df, err = load_price_data(ticker)
        if df.empty:
            rows.append({"Ticker": ticker, "Status": "No data", "Score": 0})
            continue
        df = add_indicators(df)
        latest = df.iloc[-1]
        close = safe_num(latest["Close"])
        ma20 = safe_num(latest["MA20"])
        ma50 = safe_num(latest["MA50"])
        ema9 = safe_num(latest["EMA9"])
        ema21 = safe_num(latest["EMA21"])
        rsi = safe_num(latest["RSI"], 50)
        macd_hist = safe_num(latest["MACD_HIST"], 0)

        score = 0
        if close > ma20: score += 1
        if close > ma50: score += 1
        if ema9 > ema21: score += 1
        if macd_hist > 0: score += 1
        if 45 <= rsi <= 70: score += 1
        if rsi > 75: score -= 1
        scores.append(score)
        rows.append({"Ticker": ticker, "Price": round(close, 2), "MA20": round(ma20, 2),
                     "MA50": round(ma50, 2), "RSI": round(rsi, 1), "Score": score})

    avg = np.mean(scores) if scores else 0
    if avg >= 4:
        return "🟢 Risk-On / Bullish Market", 0.40, pd.DataFrame(rows)
    if avg >= 2.5:
        return "🟡 Neutral / Mixed Market", 0.0, pd.DataFrame(rows)
    return "🔴 Risk-Off / Weak Market", -0.50, pd.DataFrame(rows)

@st.cache_data(ttl=1800)
def get_news_sentiment(ticker):
    try:
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
        r = requests.get(url, timeout=10)
        root = ET.fromstring(r.content)

        items = []
        score = 0
        for item in root.findall(".//item")[:8]:
            title = item.findtext("title", default="")
            link = item.findtext("link", default="")
            lower = title.lower()
            pos = sum(1 for w in POS_WORDS if w in lower)
            neg = sum(1 for w in NEG_WORDS if w in lower)
            s = pos - neg
            score += s
            tone = "Positive" if s > 0 else "Negative" if s < 0 else "Neutral"
            items.append({"Title": title, "Tone": tone, "Link": link})

        if score >= 2:
            return "🟢 Positive", 0.30, pd.DataFrame(items)
        if score <= -2:
            return "🔴 Negative", -0.30, pd.DataFrame(items)
        return "🟡 Neutral", 0.0, pd.DataFrame(items)
    except Exception:
        return "🟡 News unavailable", 0.0, pd.DataFrame(columns=["Title", "Tone", "Link"])

def weekly_confirmation(df):
    try:
        wdf = add_indicators(resample_weekly(df))
        latest = wdf.iloc[-1]
        close = safe_num(latest["Close"])
        ma20 = safe_num(latest["MA20"])
        ema9 = safe_num(latest["EMA9"])
        ema21 = safe_num(latest["EMA21"])
        macd_hist = safe_num(latest["MACD_HIST"])
        rsi = safe_num(latest["RSI"], 50)

        points = 0
        if close > ma20: points += 1
        if ema9 > ema21: points += 1
        if macd_hist > 0: points += 1
        if 45 <= rsi <= 70: points += 1

        if points >= 3:
            return "🟢 Weekly confirms bullish setup", 0.30, wdf
        if points == 2:
            return "🟡 Weekly is mixed", 0.0, wdf
        return "🔴 Weekly is weak", -0.30, wdf
    except Exception:
        return "🟡 Weekly unavailable", 0.0, pd.DataFrame()

WEIGHTS = {"trend": 0.30, "momentum": 0.25, "mean_rev": 0.20, "volume": 0.15, "strength": 0.10}

def score_stock(df, latest, term_type, market_adj=0.0, news_adj=0.0, mtf_adj=0.0):
    reasons = []
    close = safe_num(latest["Close"])
    ma9 = safe_num(latest["MA9"])
    ma20 = safe_num(latest["MA20"])
    ma50 = safe_num(latest["MA50"])
    ma200 = safe_num(latest["MA200"])
    ema9 = safe_num(latest["EMA9"])
    ema21 = safe_num(latest["EMA21"])
    rsi = safe_num(latest["RSI"])
    stoch_k = safe_num(latest["STOCH_K"], 50)
    stoch_d = safe_num(latest["STOCH_D"], 50)
    macd = safe_num(latest["MACD"])
    macd_sig = safe_num(latest["MACD_SIGNAL"])
    macd_hist = safe_num(latest["MACD_HIST"])
    bb_pct = safe_num(latest["BB_PCT"], 0.5)
    vol_ratio = safe_num(latest["Volume_Ratio"], 1.0)
    obv_trend = safe_num(latest["OBV_Trend"], 0)
    adx = safe_num(latest["ADX"], 20)
    plus_di = safe_num(latest["PLUS_DI"], 0)
    minus_di = safe_num(latest["MINUS_DI"], 0)
    volatility = safe_num(latest["Volatility"], 0)

    trend = 0
    if close > ma9: trend += 1.5; reasons.append("Price above MA9")
    if close > ma20: trend += 1.5; reasons.append("Price above MA20")
    if close > ma50: trend += 2.0; reasons.append("Price above MA50")
    if close > ma200: trend += 2.0 if term_type == "Long-Term" else 1.5; reasons.append("Price above long-term moving average")
    if ma20 > ma50: trend += 1.5; reasons.append("MA20 > MA50 — medium trend bullish")
    if ma50 > ma200: trend += 1.5; reasons.append("MA50 > long-term average — bullish structure")
    trend = min(trend, 10)

    momentum = 0
    if macd > macd_sig: momentum += 3; reasons.append("MACD above signal line — bullish")
    if macd_hist > 0: momentum += 2; reasons.append("MACD histogram positive — growing momentum")
    if ema9 > ema21: momentum += 3; reasons.append("EMA9 above EMA21 — short-term bullish")
    else: momentum -= 2; reasons.append("EMA9 below EMA21 — short-term bearish")
    prev_hist = safe_num(df.iloc[-2]["MACD_HIST"] if len(df) > 1 else 0)
    if macd_hist > 0 and macd_hist > prev_hist: momentum += 2; reasons.append("MACD histogram expanding — accelerating")
    momentum = max(min(momentum, 10), 0)

    mean_rev = 5
    if 45 <= rsi <= 65: mean_rev += 2; reasons.append("RSI in healthy bullish zone")
    elif rsi > 75: mean_rev -= 3; reasons.append("RSI overbought — caution")
    elif rsi < 30: mean_rev -= 2; reasons.append("RSI oversold — wait for confirmation")
    elif rsi < 45: mean_rev += 1; reasons.append("RSI near oversold — possible dip opportunity")

    if stoch_k < 20 and stoch_d < 20: mean_rev += 2; reasons.append("Stoch RSI deeply oversold — possible bounce setup")
    elif stoch_k < 30: mean_rev += 1; reasons.append("Stoch RSI oversold — possible bounce")
    elif stoch_k > 80 and stoch_d > 80: mean_rev -= 2; reasons.append("Stoch RSI deeply overbought — caution")
    elif stoch_k > 70: mean_rev -= 1; reasons.append("Stoch RSI overbought — reduce risk")

    if bb_pct < 0.15: mean_rev += 2; reasons.append("Price near lower Bollinger Band")
    elif bb_pct < 0.30: mean_rev += 1; reasons.append("Price in lower Bollinger zone")
    elif bb_pct > 0.85: mean_rev -= 1.5; reasons.append("Price near upper Bollinger Band — caution")
    mean_rev = max(min(mean_rev, 10), 0)

    volume = 5
    if obv_trend > 0: volume += 2.5; reasons.append("OBV above MA20 — money flow improving")
    else: volume -= 2; reasons.append("OBV below MA20 — money flow weakening")
    if vol_ratio > 1.5 and close > ma20: volume += 2.5; reasons.append("High volume breakout above MA20")
    elif vol_ratio > 1.2 and close > ma20: volume += 1; reasons.append("Above-average volume with bullish price")
    elif vol_ratio < 0.6: volume -= 1.5; reasons.append("Very low volume — weak conviction")
    elif vol_ratio > 1.5 and close < ma20: volume -= 2; reasons.append("High volume selling below MA20")
    volume = max(min(volume, 10), 0)

    strength = 5
    if adx >= 30 and plus_di > minus_di: strength += 4; reasons.append(f"Very strong uptrend, ADX {adx:.1f}")
    elif adx >= 25 and plus_di > minus_di: strength += 2.5; reasons.append(f"Strong uptrend, ADX {adx:.1f}")
    elif adx >= 20 and plus_di > minus_di: strength += 1; reasons.append(f"Moderate uptrend, ADX {adx:.1f}")
    elif adx < 20: strength -= 2; reasons.append(f"Weak ADX {adx:.1f} — choppy market")
    elif adx >= 25 and minus_di > plus_di: strength -= 3; reasons.append(f"Strong downtrend, ADX {adx:.1f}")
    strength = max(min(strength, 10), 0)

    raw_score = trend*0.30 + momentum*0.25 + mean_rev*0.20 + volume*0.15 + strength*0.10
    raw_score += market_adj + news_adj + mtf_adj

    if market_adj > 0: reasons.append("Market direction is supportive")
    elif market_adj < 0: reasons.append("Market direction is weak/risk-off")
    if news_adj > 0: reasons.append("Recent news sentiment is positive")
    elif news_adj < 0: reasons.append("Recent news sentiment is negative")
    if mtf_adj > 0: reasons.append("Weekly timeframe confirms daily trend")
    elif mtf_adj < 0: reasons.append("Weekly timeframe does not confirm daily trend")

    if volatility > 0.65:
        risk = "High"; raw_score *= 0.80; reasons.append("High volatility — score penalized")
    elif volatility > 0.35:
        risk = "Medium"
    else:
        risk = "Low"

    raw_score = max(min(raw_score, 10), 0)
    subscores = {
        "Trend (30%)": round(trend, 1),
        "Momentum (25%)": round(momentum, 1),
        "Mean Reversion (20%)": round(mean_rev, 1),
        "Volume / OBV (15%)": round(volume, 1),
        "Trend Strength (10%)": round(strength, 1),
        "Market Adj": round(market_adj, 2),
        "News Adj": round(news_adj, 2),
        "Multi-Timeframe Adj": round(mtf_adj, 2),
        "Weighted Total": round(raw_score, 2),
    }
    return round(raw_score, 2), risk, reasons, subscores

def estimate_future_price(df, days):
    recent = df.tail(100).copy()
    if len(recent) < 30:
        return pd.DataFrame(), 0, "Not enough data"
    log_p = np.log(recent["Close"].values)
    x = np.arange(len(log_p))
    slope, intercept = np.polyfit(x, log_p, 1)
    last_price = recent["Close"].iloc[-1]
    future_log = intercept + slope * (len(log_p) - 1 + days)
    base_price = np.exp(future_log)
    residuals = log_p - (intercept + slope * x)
    uncertainty = np.std(residuals) * np.sqrt(days)
    bull = base_price * np.exp(uncertainty * 0.5)
    bear = base_price * np.exp(-uncertainty * 0.5)
    er = (base_price / last_price) - 1
    label = ("Strong Positive" if er >= 0.08 else "Positive" if er >= 0.03 else
             "Strong Negative" if er <= -0.08 else "Negative" if er <= -0.03 else "Neutral")
    return pd.DataFrame({
        "Horizon": [f"{days} days"], "Current": [round(last_price, 2)],
        "Base Estimate": [round(base_price, 2)], "Bull Case": [round(bull, 2)],
        "Bear Case": [round(bear, 2)], "Est. Return": [f"{er:.2%}"],
        "Label": [label], "Method": ["Log-linear regression"],
    }), er, label

def final_signal(score, risk, er, term_type):
    if risk == "High" and score < 5.5: return "⚠️ Avoid / High Risk"
    if term_type == "Short-Term":
        if score >= 7.5 and er > 0: return "🔥 Strong Buy"
        if score >= 6.0 and er > 0: return "✅ Buy Signal"
        if score >= 4.5: return "📉 Buy on Dip"
        if score >= 3.5: return "⏳ Hold / Wait"
        return "🔻 Sell / High Caution"
    if score >= 8.0 and er > 0: return "🚀 Strong Long-Term Buy"
    if score >= 6.5 and er > 0: return "✅ Long-Term Buy"
    if score >= 5.0: return "📉 Long-Term Buy on Dip"
    if score >= 3.5: return "⏳ Long-Term Hold / Watch"
    return "⚠️ Avoid Long-Term"

def confidence_score(score, risk, er, adx, vol_ratio, obv_trend):
    c = 35 + score * 5
    if er > 0.05: c += 5
    elif er < -0.05: c -= 5
    if risk == "Low": c += 5
    elif risk == "High": c -= 15
    if adx >= 30: c += 8
    elif adx >= 25: c += 4
    elif adx < 20: c -= 8
    if vol_ratio > 1.5: c += 5
    elif vol_ratio < 0.6: c -= 5
    if obv_trend > 0: c += 4
    else: c -= 3
    return int(max(30, min(95, c)))

def trade_plan(latest, signal, confidence, er, horizon_days):
    close = safe_num(latest["Close"])
    atr = safe_num(latest["ATR"], close * 0.02)
    sup = safe_num(latest["Support"], close - atr)
    res = safe_num(latest["Resistance"], close + atr)
    if "Strong" in signal or "Buy Signal" in signal or "Long-Term Buy" in signal:
        buy_low = max(sup, close - 0.7 * atr); buy_high = min(close + 0.25 * atr, close * 1.015)
        target = max(res, close + 2.0 * atr); stop_loss = buy_low - 1.0 * atr; action = "🟢 BUY SETUP"
    elif "Buy on Dip" in signal:
        buy_low = max(sup, close - 1.2 * atr); buy_high = close - 0.3 * atr
        target = close + 1.8 * atr; stop_loss = buy_low - 1.0 * atr; action = "🟡 BUY ON DIP"
    elif "Sell" in signal or "Avoid" in signal:
        buy_low = np.nan; buy_high = np.nan
        target = min(sup, close - 1.5 * atr); stop_loss = close + 1.0 * atr; action = "🔴 AVOID / SELL"
    else:
        buy_low = close - 0.6 * atr; buy_high = close + 0.2 * atr
        target = close + 1.2 * atr; stop_loss = close - 1.0 * atr; action = "🟠 HOLD / WAIT"
    reward = abs(target - close); risk_amt = abs(close - stop_loss)
    rr = round(reward / risk_amt, 2) if risk_amt > 0 else 0
    hold = "1-5 days" if horizon_days <= 5 else "5-14 days" if horizon_days <= 14 else "2-8 weeks" if horizon_days <= 60 else "2-6 months"
    return pd.DataFrame({
        "Action": [action], "Buy Zone Low": [round(buy_low, 2) if not pd.isna(buy_low) else "—"],
        "Buy Zone High": [round(buy_high, 2) if not pd.isna(buy_high) else "—"],
        "Target": [round(target, 2)], "Stop Loss": [round(stop_loss, 2)],
        "Risk/Reward": [f"1 : {rr}"], "Expected Hold": [hold],
        "Confidence": [f"{confidence}%"], "Est. Return": [f"{er:.2%}"]
    })

def make_price_chart(df, ticker, signals, show_sr=True):
    tail = df.tail(126)
    latest = df.iloc[-1]
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=tail["Date"], open=tail["Open"], high=tail["High"],
                                 low=tail["Low"], close=tail["Close"], name="Price", showlegend=False))
    for col, dash in [("MA20","solid"),("MA50","dot"),("MA200","solid"),("EMA9","dash"),("EMA21","dash")]:
        fig.add_trace(go.Scatter(x=tail["Date"], y=tail[col], mode="lines", name=col, line=dict(width=1.2, dash=dash)))
    if show_sr:
        support = safe_num(latest["Support"]); resistance = safe_num(latest["Resistance"])
        fig.add_hline(y=support, line_dash="dash", annotation_text=f"Support {support:.2f}")
        fig.add_hline(y=resistance, line_dash="dash", annotation_text=f"Resistance {resistance:.2f}")
    sig_df = pd.DataFrame(signals) if signals else pd.DataFrame()
    if not sig_df.empty:
        sig_df = sig_df[sig_df["Date"] >= tail["Date"].min()]
        buys = sig_df[sig_df["Type"] == "BUY"]
        sells = sig_df[sig_df["Type"] == "SELL"]
        if not buys.empty:
            fig.add_trace(go.Scatter(x=buys["Date"], y=buys["Price"]*0.985, mode="markers+text",
                                     marker=dict(symbol="triangle-up", size=14), text=["B"]*len(buys),
                                     textposition="bottom center", name="BUY Signal", hovertext=buys["Reason"]))
        if not sells.empty:
            fig.add_trace(go.Scatter(x=sells["Date"], y=sells["Price"]*1.015, mode="markers+text",
                                     marker=dict(symbol="triangle-down", size=14), text=["S"]*len(sells),
                                     textposition="top center", name="SELL Signal", hovertext=sells["Reason"]))
    fig.update_layout(title=f"{ticker} — Price + Buy/Sell Signals + Support/Resistance", height=560,
                      template="plotly_dark", xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=-0.15))
    return fig

def make_line_chart(df, ticker, cols, title):
    tail = df.tail(126)
    fig = go.Figure()
    for col in cols:
        fig.add_trace(go.Scatter(x=tail["Date"], y=tail[col], mode="lines", name=col))
    fig.update_layout(title=f"{ticker} {title}", height=260, template="plotly_dark")
    return fig

def make_volume_chart(df, ticker):
    tail = df.tail(126)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=tail["Date"], y=tail["Volume"], name="Volume"))
    fig.add_trace(go.Scatter(x=tail["Date"], y=tail["Volume_MA20"], mode="lines", name="Vol MA20"))
    fig.update_layout(title=f"{ticker} Volume", height=260, template="plotly_dark")
    return fig

def make_weekly_chart(wdf, ticker):
    tail = wdf.tail(52)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=tail["Date"], open=tail["Open"], high=tail["High"],
                                 low=tail["Low"], close=tail["Close"], name="Weekly"))
    for col in ["MA20", "MA50", "EMA9", "EMA21"]:
        if col in tail.columns:
            fig.add_trace(go.Scatter(x=tail["Date"], y=tail[col], mode="lines", name=col))
    fig.update_layout(title=f"{ticker} Weekly Timeframe", height=420, template="plotly_dark", xaxis_rangeslider_visible=False)
    return fig

def signal_box(signal):
    if "Strong" in signal or "Buy Signal" in signal or "Long-Term Buy" in signal:
        st.success(f"## {signal}")
    elif "Buy on Dip" in signal:
        st.warning(f"## {signal}")
    elif "Hold" in signal or "Wait" in signal:
        st.info(f"## {signal}")
    else:
        st.error(f"## {signal}")

def analyze_stock(ticker, horizon_days, term_type, include_news=True, include_market=True):
    df, err = load_price_data(ticker)
    if df.empty: return None, err
    df = add_indicators(df)
    if df.empty: return None, "Not enough data after indicators."

    market_label, market_adj, market_df = get_market_direction() if include_market else ("Not used", 0.0, pd.DataFrame())
    news_label, news_adj, news_df = get_news_sentiment(ticker) if include_news else ("Not used", 0.0, pd.DataFrame())
    mtf_label, mtf_adj, weekly_df = weekly_confirmation(df)

    latest = df.iloc[-1]
    signals = detect_buysell_signals(df)
    score, risk, reasons, subscores = score_stock(df, latest, term_type, market_adj, news_adj, mtf_adj)
    est_df, er, label = estimate_future_price(df, horizon_days)
    adx = safe_num(latest["ADX"], 20); vol_ratio = safe_num(latest["Volume_Ratio"], 1.0); obv_trend = safe_num(latest["OBV_Trend"], 0)
    sig = final_signal(score, risk, er, term_type)
    conf = confidence_score(score, risk, er, adx, vol_ratio, obv_trend)
    plan = trade_plan(latest, sig, conf, er, horizon_days)
    last_signal = signals[-1] if signals else None

    return {
        "df": df, "weekly_df": weekly_df, "latest": latest, "score": score, "subscores": subscores,
        "risk": risk, "reasons": reasons, "signals": signals, "last_signal": last_signal,
        "est_df": est_df, "er": er, "label": label, "signal": sig, "confidence": conf,
        "plan": plan, "adx": adx, "vol_ratio": vol_ratio, "obv_trend": obv_trend,
        "bb_pct": safe_num(latest["BB_PCT"], 0.5), "rsi": safe_num(latest["RSI"], 50),
        "stoch_k": safe_num(latest["STOCH_K"], 50), "market_label": market_label,
        "market_df": market_df, "news_label": news_label, "news_df": news_df,
        "mtf_label": mtf_label, "support": safe_num(latest["Support"]), "resistance": safe_num(latest["Resistance"])
    }, ""

def scan_parallel(tickers, horizon_days, term_type, max_workers=2):
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(analyze_stock, t, horizon_days, term_type, False, True): t for t in tickers}
        for future in as_completed(futures):
            t = futures[future]
            try:
                res, err = future.result(timeout=45)
                results[t] = (res, err)
            except Exception as e:
                results[t] = (None, str(e))
    return results

st.sidebar.header("⚙️ Settings")
term_type = st.sidebar.radio("Trading Style", ["Short-Term", "Long-Term"])
selected_ticker = st.sidebar.selectbox("Select Stock", ALL_TICKERS, index=ALL_TICKERS.index("NVDA") if "NVDA" in ALL_TICKERS else 0)
horizon_days = st.sidebar.selectbox("Forecast Horizon (days)", [1,3,5,10,14,30,60,90], index=2)
scan_count = st.sidebar.selectbox("Scanner — how many stocks?", [5,10,25], index=0)
include_news = st.sidebar.checkbox("Include news sentiment", value=True)
include_market = st.sidebar.checkbox("Include market direction", value=True)
show_sr = st.sidebar.checkbox("Show support/resistance lines", value=True)
st.sidebar.info("If Yahoo shows Too Many Requests, wait 10-30 minutes, keep scanner at 5, and avoid repeated refreshes.")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Single Stock", "🔍 Scanner", "🌎 Market", "📋 Ticker List"])

with tab1:
    st.subheader(f"Analysis — {selected_ticker}")
    with st.spinner(f"Loading {selected_ticker}…"):
        result, error = analyze_stock(selected_ticker, horizon_days, term_type, include_news, include_market)

    if result is None:
        st.error(f"Could not analyse {selected_ticker}: {error}")
    else:
        latest = result["latest"]
        c1, c2, c3 = st.columns(3)
        c1.info(f"Market: {result['market_label']}")
        c2.info(f"News: {result['news_label']}")
        c3.info(f"Weekly: {result['mtf_label']}")

        ls = result["last_signal"]
        if ls:
            msg = f"Last auto-signal: {ls['Type']} on {pd.Timestamp(ls['Date']).strftime('%b %d %Y')} @ ${ls['Price']:.2f} — {ls['Reason']}"
            st.success(msg) if ls["Type"] == "BUY" else st.error(msg)

        cols = st.columns(10)
        metrics = [
            ("Price", f"${safe_num(latest['Close']):.2f}"), ("Signal", result["signal"][:18]),
            ("Score", f"{result['score']:.1f}/10"), ("Confidence", f"{result['confidence']}%"),
            ("Risk", result["risk"]), ("ADX", f"{result['adx']:.1f}"), ("RSI", f"{result['rsi']:.1f}"),
            ("Stoch K", f"{result['stoch_k']:.1f}"), ("Support", f"${result['support']:.2f}"),
            ("Resistance", f"${result['resistance']:.2f}")
        ]
        for col, (label, val) in zip(cols, metrics):
            col.metric(label, val)

        signal_box(result["signal"])
        st.markdown("### 📋 Trade Plan")
        st.dataframe(result["plan"], width="stretch")

        st.markdown("### 📈 Price Chart + Buy / Sell Signals")
        st.plotly_chart(make_price_chart(result["df"], selected_ticker, result["signals"], show_sr), width="stretch")

        if result["signals"]:
            st.markdown("### 🔔 Historical Buy/Sell Signals")
            sig_df = pd.DataFrame(result["signals"])
            sig_df["Date"] = pd.to_datetime(sig_df["Date"]).dt.strftime("%Y-%m-%d")
            sig_df["Price"] = sig_df["Price"].round(2)
            st.dataframe(sig_df, width="stretch", height=280)

        r1c1, r1c2 = st.columns(2)
        with r1c1:
            st.plotly_chart(make_line_chart(result["df"], selected_ticker, ["MACD","MACD_SIGNAL","MACD_HIST"], "MACD"), width="stretch")
        with r1c2:
            st.plotly_chart(make_line_chart(result["df"], selected_ticker, ["RSI","STOCH_K","STOCH_D"], "RSI + Stochastic RSI"), width="stretch")

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            st.plotly_chart(make_line_chart(result["df"], selected_ticker, ["OBV","OBV_MA20"], "OBV"), width="stretch")
        with r2c2:
            st.plotly_chart(make_volume_chart(result["df"], selected_ticker), width="stretch")

        st.plotly_chart(make_line_chart(result["df"], selected_ticker, ["ADX","PLUS_DI","MINUS_DI"], "ADX Trend Strength"), width="stretch")

        if not result["weekly_df"].empty:
            st.markdown("### 🗓️ Weekly Timeframe Confirmation")
            st.plotly_chart(make_weekly_chart(result["weekly_df"], selected_ticker), width="stretch")

        st.markdown("### 📰 News Sentiment")
        st.dataframe(result["news_df"], width="stretch", height=280)

        st.markdown("### 🔮 Price Forecast")
        st.dataframe(result["est_df"], width="stretch")

        st.markdown("### 🧮 Score Breakdown")
        sb = result["subscores"]
        st.dataframe(pd.DataFrame([{"Factor": k, "Score /10": v} for k, v in sb.items()]), width="stretch")

        st.markdown("### 💬 Signal Reasoning")
        for r in result["reasons"]:
            st.write(f"- {r}")

with tab2:
    st.subheader("⚡ Scanner + Buckets")
    scan_tickers = ALL_TICKERS[:scan_count]
    if st.button("▶️ Run Scanner", type="primary"):
        with st.spinner("Scanning…"):
            raw_results = scan_parallel(scan_tickers, horizon_days, term_type, max_workers=2)

        rows, failed = [], []
        for ticker in scan_tickers:
            res, err = raw_results.get(ticker, (None, "timeout"))
            if res is None:
                failed.append({"Ticker": ticker, "Error": err})
                continue
            latest = res["latest"]
            rows.append({
                "Ticker": ticker, "Price": round(safe_num(latest["Close"]), 2),
                "Signal": res["signal"], "Score": res["score"], "Confidence": res["confidence"],
                "Est. Return %": round(res["er"] * 100, 2), "Risk": res["risk"],
                "ADX": round(res["adx"], 1), "RSI": round(res["rsi"], 1),
                "Stoch K": round(res["stoch_k"], 1),
                "OBV Trend": "↑ Bullish" if res["obv_trend"] > 0 else "↓ Bearish",
                "Vol Ratio": round(res["vol_ratio"], 2), "BB%": round(res["bb_pct"], 2),
                "Support": round(res["support"], 2), "Resistance": round(res["resistance"], 2),
                "Market": res["market_label"], "Weekly": res["mtf_label"]
            })

        if rows:
            sdf = pd.DataFrame(rows).sort_values(["Score","Confidence"], ascending=False)
            st.success(f"✅ Scanned {len(sdf)} stocks")
            st.dataframe(sdf, width="stretch", height=450)

            st.markdown("### 🏆 Strongest Momentum")
            st.dataframe(sdf.sort_values(["Score","ADX","Confidence"], ascending=False).head(10), width="stretch")

            st.markdown("### 🟡 Oversold Bounce Setups")
            bounce = sdf[(sdf["RSI"] <= 45) | (sdf["Stoch K"] <= 30)].sort_values(["Confidence","Score"], ascending=False)
            st.dataframe(bounce.head(10), width="stretch")

            st.markdown("### 🚀 Breakout Candidates")
            breakout = sdf[(sdf["Vol Ratio"] >= 1.2) & (sdf["ADX"] >= 20)].sort_values(["Score","Vol Ratio"], ascending=False)
            st.dataframe(breakout.head(10), width="stretch")

            st.download_button("⬇️ Download Results CSV", sdf.to_csv(index=False).encode("utf-8"), "scanner_v4.csv", "text/csv")

        if failed:
            with st.expander("Show failed tickers"):
                st.dataframe(pd.DataFrame(failed), width="stretch")
        if not rows:
            st.warning("No results returned. Yahoo may be rate-limiting your app.")
    else:
        st.info("Press Run Scanner. Keep scanner at 5 if you get Yahoo rate-limit errors.")

with tab3:
    st.subheader("🌎 Market Direction")
    label, adj, market_df = get_market_direction()
    st.info(f"{label} | Score adjustment used in model: {adj}")
    st.dataframe(market_df, width="stretch")

with tab4:
    st.subheader("📋 Ticker List")
    st.write(f"Total tickers loaded: **{len(ALL_TICKERS)}**")
    st.dataframe(pd.DataFrame({"Ticker": ALL_TICKERS}), width="stretch", height=700)

