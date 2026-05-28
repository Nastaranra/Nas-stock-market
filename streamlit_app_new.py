import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import xml.etree.ElementTree as ET
import plotly.graph_objects as go
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(page_title="AI Trading Signal App v4", layout="wide", page_icon="📈")

st.title("📈 AI Trading Signal App v4")
st.caption("Market Direction · News Sentiment · Support/Resistance · Multi-Timeframe · Scanner Ranking")
st.warning("⚠️ Educational only. Not financial advice. Past signals do not guarantee future results.")

DEFAULT_TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","GOOG","META","TSLA","NFLX","AMD",
    "SPY","QQQ","PLTR","SMCI","AVGO","CRM","MU","INTC","UBER","SHOP",
    "COIN","SOFI","PYPL","ADBE","PANW","MSTR","ARM","BABA","DIS","NKE",
    "COST","WMT","TGT","JPM","BAC","V","MA","UNH","LLY","XOM","CVX",
    "HOOD","RIVN","LCID","NIO","F","GM","BA","GE","T","VZ","PFE",
    "MRNA","KO","PEP","SBUX","ORCL","IBM"
]

POSITIVE_WORDS = [
    "beat","beats","surge","surges","jump","jumps","rally","rallies",
    "upgrade","upgraded","bullish","growth","record","strong","raises",
    "outperform","buy","positive","profit","partnership","expands","higher",
    "ai demand","revenue growth","record high","new high"
]

NEGATIVE_WORDS = [
    "miss","misses","fall","falls","drop","drops","plunge","plunges",
    "downgrade","downgraded","bearish","weak","cuts","lawsuit",
    "investigation","warning","loss","negative","sell","concern",
    "slows","slowing","tariff","risk","lower","probe","underperform"
]

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

def clean_price_data(df, ticker):
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
        df, err = clean_price_data(raw, ticker)
        if not df.empty:
            return df, ""

        return pd.DataFrame(), (
            f"No data returned for {ticker}. Yahoo may be rate-limiting requests. "
            "Wait 10-30 minutes, reduce scanner count, or try again later."
        )

    except Exception as e:
        msg = str(e)
        if "Too Many Requests" in msg or "Rate limited" in msg or "429" in msg:
            return pd.DataFrame(), (
                "Yahoo Finance rate limit reached. Wait 10-30 minutes and try again. "
                "Use Scanner count = 5 and avoid refreshing repeatedly."
            )
        return pd.DataFrame(), msg

@st.cache_data(ttl=3600)
def get_news_sentiment(ticker, max_items=10):
    """
    FIXED Recent News Sentiment:
    - Uses yfinance news first
    - Skips blank title rows
    - If yfinance news is empty/blank, uses Yahoo Finance RSS fallback
    - Prevents empty Neutral rows in the table
    """
    rows = []

    def score_title(title):
        text = str(title).lower()
        pos = sum(1 for w in POSITIVE_WORDS if w in text)
        neg = sum(1 for w in NEGATIVE_WORDS if w in text)

        if pos > neg:
            return "Positive", 1
        elif neg > pos:
            return "Negative", -1
        else:
            return "Neutral", 0

    # 1) Try yfinance news first
    try:
        news = yf.Ticker(ticker).news or []

        for item in news[:max_items]:
            title = item.get("title", "")
            publisher = item.get("publisher", "")
            link = item.get("link", "")
            ts = item.get("providerPublishTime", None)

            if not str(title).strip():
                continue

            dt = pd.to_datetime(ts, unit="s", errors="coerce") if ts else pd.NaT
            label, score = score_title(title)

            rows.append({
                "Date": dt.strftime("%Y-%m-%d") if not pd.isna(dt) else "",
                "Title": title,
                "Publisher": publisher if publisher else "Yahoo Finance",
                "Sentiment": label,
                "Score": score,
                "Link": link
            })

    except Exception:
        pass

    # 2) Yahoo Finance RSS fallback
    if not rows:
        try:
            rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
            r = requests.get(rss_url, timeout=10)

            if r.status_code == 200 and r.content:
                root = ET.fromstring(r.content)

                for item in root.findall(".//item")[:max_items]:
                    title = item.findtext("title", default="")
                    link = item.findtext("link", default="")
                    pub_date = item.findtext("pubDate", default="")

                    if not str(title).strip():
                        continue

                    label, score = score_title(title)

                    rows.append({
                        "Date": pub_date,
                        "Title": title,
                        "Publisher": "Yahoo Finance RSS",
                        "Sentiment": label,
                        "Score": score,
                        "Link": link
                    })

        except Exception:
            pass

    if not rows:
        return pd.DataFrame(columns=["Date", "Title", "Publisher", "Sentiment", "Score", "Link"]), 0, "No news found"

    df = pd.DataFrame(rows)

    for col in ["Date", "Title", "Publisher", "Sentiment", "Score", "Link"]:
        if col not in df.columns:
            df[col] = ""

    df = df[df["Title"].astype(str).str.strip() != ""].copy()

    if df.empty:
        return pd.DataFrame(columns=["Date", "Title", "Publisher", "Sentiment", "Score", "Link"]), 0, "No news found"

    avg = pd.to_numeric(df["Score"], errors="coerce").fillna(0).mean()

    if avg > 0.20:
        overall = "Positive"
    elif avg < -0.20:
        overall = "Negative"
    else:
        overall = "Neutral"

    return df, avg, overall

@st.cache_data(ttl=86400)
def get_earnings_date(ticker):
    try:
        cal = yf.Ticker(ticker).calendar
        if isinstance(cal, dict):
            ed = cal.get("Earnings Date")
            if ed:
                return pd.Timestamp(ed[0] if isinstance(ed, (list, tuple)) else ed).date()
        elif isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.index:
            return pd.Timestamp(cal.loc["Earnings Date"].iloc[0]).date()
    except Exception:
        pass
    return None

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
    df["BB_WIDTH"] = (df["BB_UPPER"] - df["BB_LOWER"]) / df["BB_MID"]
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

def summarize_index(ticker):
    df, err = load_price_data(ticker, period="6mo", interval="1d")
    if df.empty:
        return {"Ticker": ticker, "Label": "Unavailable", "Score": 0, "Reason": err}

    df = add_indicators(df)
    if df.empty:
        return {"Ticker": ticker, "Label": "Unavailable", "Score": 0, "Reason": "Not enough data after indicators"}

    latest = df.iloc[-1]
    close = safe_num(latest["Close"])
    ma20 = safe_num(latest["MA20"])
    ma50 = safe_num(latest["MA50"])
    ma200 = safe_num(latest["MA200"])
    rsi = safe_num(latest["RSI"], 50)

    score = 0
    reasons = []

    if close > ma20:
        score += 1
        reasons.append("above MA20")
    else:
        score -= 1
        reasons.append("below MA20")

    if close > ma50:
        score += 1
        reasons.append("above MA50")
    else:
        score -= 1
        reasons.append("below MA50")

    if ma20 > ma50:
        score += 1
        reasons.append("MA20 > MA50")
    else:
        score -= 1
        reasons.append("MA20 < MA50")

    if close > ma200:
        score += 1
        reasons.append("above long-term average")
    else:
        score -= 1
        reasons.append("below long-term average")

    if 45 <= rsi <= 70:
        score += 1
        reasons.append("healthy RSI")
    elif rsi > 75 or rsi < 40:
        score -= 1
        reasons.append("RSI caution zone")

    if score >= 3:
        label = "Bullish"
    elif score <= -2:
        label = "Bearish"
    else:
        label = "Neutral"

    return {
        "Ticker": ticker,
        "Label": label,
        "Score": score,
        "Price": round(close, 2),
        "RSI": round(rsi, 1),
        "Reason": ", ".join(reasons)
    }

@st.cache_data(ttl=1800)
def market_direction():
    spy = summarize_index("SPY")
    qqq = summarize_index("QQQ")
    vix = summarize_index("^VIX")

    score = spy.get("Score", 0) + qqq.get("Score", 0)

    vix_price = vix.get("Price", np.nan)
    vix_label = "Unavailable"

    if not pd.isna(vix_price):
        if vix_price < 16:
            vix_label = "Low Fear"
            score += 1
        elif vix_price <= 22:
            vix_label = "Normal"
        else:
            vix_label = "High Fear"
            score -= 2

    if score >= 5:
        overall = "Bullish Market"
    elif score <= 0:
        overall = "Bearish / Risk-Off Market"
    else:
        overall = "Neutral / Mixed Market"

    rows = [
        spy,
        qqq,
        {
            "Ticker": "^VIX",
            "Label": vix_label,
            "Score": vix.get("Score", 0),
            "Price": vix.get("Price", ""),
            "RSI": vix.get("RSI", ""),
            "Reason": vix.get("Reason", "")
        }
    ]

    return overall, score, pd.DataFrame(rows)

def resample_weekly(df):
    w = df.copy().set_index("Date")
    weekly = pd.DataFrame()
    weekly["Open"] = w["Open"].resample("W-FRI").first()
    weekly["High"] = w["High"].resample("W-FRI").max()
    weekly["Low"] = w["Low"].resample("W-FRI").min()
    weekly["Close"] = w["Close"].resample("W-FRI").last()
    weekly["Volume"] = w["Volume"].resample("W-FRI").sum()
    return weekly.dropna().reset_index()

def timeframe_label(df):
    if df is None or df.empty or len(df) < 30:
        return "Unavailable", 0, "Not enough data"

    df = add_indicators(df)
    if df.empty:
        return "Unavailable", 0, "Not enough data after indicators"

    latest = df.iloc[-1]

    close = safe_num(latest["Close"])
    ma20 = safe_num(latest["MA20"])
    ma50 = safe_num(latest["MA50"])
    ema9 = safe_num(latest["EMA9"])
    ema21 = safe_num(latest["EMA21"])
    macd_hist = safe_num(latest["MACD_HIST"])
    rsi = safe_num(latest["RSI"], 50)

    score = 0
    reasons = []

    if close > ma20:
        score += 1
        reasons.append("price > MA20")
    else:
        score -= 1
        reasons.append("price < MA20")

    if close > ma50:
        score += 1
        reasons.append("price > MA50")
    else:
        score -= 1
        reasons.append("price < MA50")

    if ema9 > ema21:
        score += 1
        reasons.append("EMA9 > EMA21")
    else:
        score -= 1
        reasons.append("EMA9 < EMA21")

    if macd_hist > 0:
        score += 1
        reasons.append("MACD positive")
    else:
        score -= 1
        reasons.append("MACD negative")

    if 45 <= rsi <= 70:
        score += 1
        reasons.append("healthy RSI")
    elif rsi > 75 or rsi < 40:
        score -= 1
        reasons.append("RSI caution zone")

    if score >= 3:
        label = "Bullish"
    elif score <= -2:
        label = "Bearish"
    else:
        label = "Neutral"

    return label, score, ", ".join(reasons)

def multi_timeframe_confirmation(df):
    try:
        daily_label, daily_score, daily_reason = timeframe_label(df)
    except Exception as e:
        daily_label, daily_score, daily_reason = "Unavailable", 0, str(e)

    try:
        weekly_df = resample_weekly(df)
        weekly_label, weekly_score, weekly_reason = timeframe_label(weekly_df)
    except Exception as e:
        weekly_label, weekly_score, weekly_reason = "Unavailable", 0, str(e)

    if daily_label == "Bullish" and weekly_label == "Bullish":
        final = "Strong Bullish Alignment"
    elif daily_label == "Bearish" and weekly_label == "Bearish":
        final = "Strong Bearish Alignment"
    elif daily_label == "Bullish" and weekly_label in ["Neutral", "Bullish"]:
        final = "Bullish, but confirm entry"
    elif daily_label == "Bearish" and weekly_label in ["Neutral", "Bearish"]:
        final = "Short-term weakness"
    else:
        final = "Mixed / Wait for confirmation"

    return pd.DataFrame([
        {"Timeframe": "Daily", "Label": daily_label, "Score": daily_score, "Reason": daily_reason},
        {"Timeframe": "Weekly", "Label": weekly_label, "Score": weekly_score, "Reason": weekly_reason},
        {"Timeframe": "Overall", "Label": final, "Score": daily_score + weekly_score, "Reason": ""}
    ]), final

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
            signals.append({
                "Date": row["Date"],
                "Price": row["Close"],
                "Type": "BUY",
                "Reason": "EMA9 crossed above EMA21 + MACD bullish + OBV inflow"
            })
            continue

        ema_cross_dn = (prev_diff >= 0) and (ema_diff < 0)
        double_ob = (row["RSI"] > 70) and (row["STOCH_K"] > 80)
        macd_bearish = (row["MACD_HIST"] < 0) and (prev_row["MACD_HIST"] >= 0)

        if ema_cross_dn or double_ob or macd_bearish:
            signals.append({
                "Date": row["Date"],
                "Price": row["Close"],
                "Type": "SELL",
                "Reason": (
                    "EMA cross down" if ema_cross_dn else
                    "RSI+Stoch double overbought" if double_ob else
                    "MACD histogram turned negative"
                )
            })

    return signals

WEIGHTS = {
    "trend": 0.30,
    "momentum": 0.25,
    "mean_rev": 0.20,
    "volume": 0.15,
    "strength": 0.10,
}

def score_stock(df, latest, term_type, market_label, mtf_final, news_overall):
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
    if close > ma9:
        trend += 1.5
        reasons.append("Price above MA9")
    if close > ma20:
        trend += 1.5
        reasons.append("Price above MA20")
    if close > ma50:
        trend += 2.0
        reasons.append("Price above MA50")
    if close > ma200:
        trend += 2.0 if term_type == "Long-Term" else 1.5
        reasons.append("Price above long-term moving average")
    if ma20 > ma50:
        trend += 1.5
        reasons.append("MA20 > MA50 — medium trend bullish")
    if ma50 > ma200:
        trend += 1.5
        reasons.append("MA50 > long-term average — bullish structure")
    trend = min(trend, 10)

    momentum = 0
    if macd > macd_sig:
        momentum += 3.0
        reasons.append("MACD above signal line — bullish")
    if macd_hist > 0:
        momentum += 2.0
        reasons.append("MACD histogram positive — growing momentum")
    if ema9 > ema21:
        momentum += 3.0
        reasons.append("EMA9 above EMA21 — short-term bullish")
    else:
        momentum -= 2.0
        reasons.append("EMA9 below EMA21 — short-term bearish")

    prev_hist = safe_num(df.iloc[-2]["MACD_HIST"] if len(df) > 1 else 0, 0)
    if macd_hist > 0 and macd_hist > prev_hist:
        momentum += 2.0
        reasons.append("MACD histogram expanding — accelerating")
    momentum = max(min(momentum, 10), 0)

    mean_rev = 5.0
    if 45 <= rsi <= 65:
        mean_rev += 2.0
        reasons.append("RSI in healthy bullish zone")
    elif rsi > 75:
        mean_rev -= 3.0
        reasons.append("RSI overbought — caution")
    elif rsi < 30:
        mean_rev -= 2.0
        reasons.append("RSI oversold — wait for confirmation")
    elif rsi < 45:
        mean_rev += 1.0
        reasons.append("RSI near oversold — possible dip opportunity")

    if stoch_k < 20 and stoch_d < 20:
        mean_rev += 2.0
        reasons.append("Stoch RSI deeply oversold — possible bounce setup")
    elif stoch_k < 30:
        mean_rev += 1.0
        reasons.append("Stoch RSI oversold — possible bounce")
    elif stoch_k > 80 and stoch_d > 80:
        mean_rev -= 2.0
        reasons.append("Stoch RSI deeply overbought — caution")
    elif stoch_k > 70:
        mean_rev -= 1.0
        reasons.append("Stoch RSI overbought — reduce risk")

    if bb_pct < 0.15:
        mean_rev += 2.0
        reasons.append("Price near lower Bollinger Band")
    elif bb_pct < 0.30:
        mean_rev += 1.0
        reasons.append("Price in lower Bollinger zone")
    elif bb_pct > 0.85:
        mean_rev -= 1.5
        reasons.append("Price near upper Bollinger Band — caution")
    mean_rev = max(min(mean_rev, 10), 0)

    volume = 5.0
    if obv_trend > 0:
        volume += 2.5
        reasons.append("OBV above MA20 — money flow improving")
    else:
        volume -= 2.0
        reasons.append("OBV below MA20 — money flow weakening")

    if vol_ratio > 1.5 and close > ma20:
        volume += 2.5
        reasons.append("High volume breakout above MA20")
    elif vol_ratio > 1.2 and close > ma20:
        volume += 1.0
        reasons.append("Above-average volume with bullish price")
    elif vol_ratio < 0.6:
        volume -= 1.5
        reasons.append("Very low volume — weak conviction")
    elif vol_ratio > 1.5 and close < ma20:
        volume -= 2.0
        reasons.append("High volume selling below MA20")
    volume = max(min(volume, 10), 0)

    strength = 5.0
    if adx >= 30 and plus_di > minus_di:
        strength += 4.0
        reasons.append(f"Very strong uptrend, ADX {adx:.1f}")
    elif adx >= 25 and plus_di > minus_di:
        strength += 2.5
        reasons.append(f"Strong uptrend, ADX {adx:.1f}")
    elif adx >= 20 and plus_di > minus_di:
        strength += 1.0
        reasons.append(f"Moderate uptrend, ADX {adx:.1f}")
    elif adx < 20:
        strength -= 2.0
        reasons.append(f"Weak ADX {adx:.1f} — choppy market")
    elif adx >= 25 and minus_di > plus_di:
        strength -= 3.0
        reasons.append(f"Strong downtrend, ADX {adx:.1f}")
    strength = max(min(strength, 10), 0)

    raw_score = (
        trend * WEIGHTS["trend"] +
        momentum * WEIGHTS["momentum"] +
        mean_rev * WEIGHTS["mean_rev"] +
        volume * WEIGHTS["volume"] +
        strength * WEIGHTS["strength"]
    )

    if "Bullish Market" in market_label:
        raw_score += 0.25
        reasons.append("Market direction bullish — small score boost")
    elif "Bearish" in market_label:
        raw_score -= 0.40
        reasons.append("Market direction bearish/risk-off — score penalty")

    if "Strong Bullish" in mtf_final:
        raw_score += 0.35
        reasons.append("Daily + weekly alignment bullish — score boost")
    elif "Strong Bearish" in mtf_final:
        raw_score -= 0.50
        reasons.append("Daily + weekly alignment bearish — score penalty")
    elif "Mixed" in mtf_final:
        raw_score -= 0.15
        reasons.append("Multi-timeframe mixed — wait for confirmation")

    if news_overall == "Positive":
        raw_score += 0.20
        reasons.append("Recent news sentiment positive — small boost")
    elif news_overall == "Negative":
        raw_score -= 0.25
        reasons.append("Recent news sentiment negative — small penalty")

    if volatility > 0.65:
        risk = "High"
        raw_score *= 0.80
        reasons.append("High volatility — score penalized")
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
        "v4 Adjusted Total": round(raw_score, 2),
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

    label = ("Strong Positive" if er >= 0.08 else
             "Positive" if er >= 0.03 else
             "Strong Negative" if er <= -0.08 else
             "Negative" if er <= -0.03 else "Neutral")

    out = pd.DataFrame({
        "Horizon": [f"{days} days"],
        "Current": [round(last_price, 2)],
        "Base Estimate": [round(base_price, 2)],
        "Bull Case": [round(bull, 2)],
        "Bear Case": [round(bear, 2)],
        "Est. Return": [f"{er:.2%}"],
        "Label": [label],
        "Method": ["Log-linear regression"],
    })
    return out, er, label

def final_signal(score, risk, er, term_type, mtf_final):
    if risk == "High" and score < 5.5:
        return "⚠️ Avoid / High Risk"

    if "Strong Bearish" in mtf_final and score < 6.5:
        return "🔻 Sell / High Caution"

    if term_type == "Short-Term":
        if score >= 7.5 and er > 0:
            return "🔥 Strong Buy"
        if score >= 6.0 and er > 0:
            return "✅ Buy Signal"
        if score >= 4.5:
            return "📉 Buy on Dip"
        if score >= 3.5:
            return "⏳ Hold / Wait"
        return "🔻 Sell / High Caution"

    if score >= 8.0 and er > 0:
        return "🚀 Strong Long-Term Buy"
    if score >= 6.5 and er > 0:
        return "✅ Long-Term Buy"
    if score >= 5.0:
        return "📉 Long-Term Buy on Dip"
    if score >= 3.5:
        return "⏳ Long-Term Hold / Watch"
    return "⚠️ Avoid Long-Term"

def confidence_score(score, risk, er, adx, vol_ratio, obv_trend, market_label, mtf_final, news_overall):
    c = 35 + score * 5

    if er > 0.05:
        c += 5
    elif er < -0.05:
        c -= 5

    if risk == "Low":
        c += 5
    elif risk == "High":
        c -= 15

    if adx >= 30:
        c += 8
    elif adx >= 25:
        c += 4
    elif adx < 20:
        c -= 8

    if vol_ratio > 1.5:
        c += 5
    elif vol_ratio < 0.6:
        c -= 5

    if obv_trend > 0:
        c += 4
    else:
        c -= 3

    if "Bullish Market" in market_label:
        c += 3
    elif "Bearish" in market_label:
        c -= 5

    if "Strong Bullish" in mtf_final:
        c += 5
    elif "Strong Bearish" in mtf_final:
        c -= 8

    if news_overall == "Positive":
        c += 2
    elif news_overall == "Negative":
        c -= 3

    return int(max(30, min(95, c)))

def trade_plan(latest, signal, confidence, er, horizon_days):
    close = safe_num(latest["Close"])
    atr = safe_num(latest["ATR"], close * 0.02)
    sup = safe_num(latest["Support"], close - atr)
    res = safe_num(latest["Resistance"], close + atr)

    if "Strong" in signal or "Buy Signal" in signal or "Long-Term Buy" in signal:
        buy_low = max(sup, close - 0.7 * atr)
        buy_high = min(close + 0.25 * atr, close * 1.015)
        target = max(res, close + 2.0 * atr)
        stop_loss = buy_low - 1.0 * atr
        action = "🟢 BUY SETUP"
    elif "Buy on Dip" in signal:
        buy_low = max(sup, close - 1.2 * atr)
        buy_high = close - 0.3 * atr
        target = close + 1.8 * atr
        stop_loss = buy_low - 1.0 * atr
        action = "🟡 BUY ON DIP"
    elif "Sell" in signal or "Avoid" in signal:
        buy_low = np.nan
        buy_high = np.nan
        target = min(sup, close - 1.5 * atr)
        stop_loss = close + 1.0 * atr
        action = "🔴 AVOID / SELL"
    else:
        buy_low = close - 0.6 * atr
        buy_high = close + 0.2 * atr
        target = close + 1.2 * atr
        stop_loss = close - 1.0 * atr
        action = "🟠 HOLD / WAIT"

    reward = abs(target - close)
    risk_amt = abs(close - stop_loss)
    rr = round(reward / risk_amt, 2) if risk_amt > 0 else 0

    hold = ("1-5 days" if horizon_days <= 5 else
            "5-14 days" if horizon_days <= 14 else
            "2-8 weeks" if horizon_days <= 60 else
            "2-6 months")

    return pd.DataFrame({
        "Action": [action],
        "Buy Zone Low": [round(buy_low, 2) if not pd.isna(buy_low) else "—"],
        "Buy Zone High": [round(buy_high, 2) if not pd.isna(buy_high) else "—"],
        "Target": [round(target, 2)],
        "Stop Loss": [round(stop_loss, 2)],
        "Risk/Reward": [f"1 : {rr}"],
        "Expected Hold": [hold],
        "Confidence": [f"{confidence}%"],
        "Est. Return": [f"{er:.2%}"],
    })

def make_price_chart(df, ticker, signals, show_sr=True):
    tail = df.tail(126)
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=tail["Date"], open=tail["Open"], high=tail["High"],
        low=tail["Low"], close=tail["Close"], name="Price", showlegend=False
    ))

    for col, dash in [
        ("MA20", "solid"), ("MA50", "dot"), ("MA200", "solid"),
        ("EMA9", "dash"), ("EMA21", "dash")
    ]:
        fig.add_trace(go.Scatter(
            x=tail["Date"], y=tail[col], mode="lines", name=col,
            line=dict(width=1.2, dash=dash)
        ))

    if show_sr and not tail.empty:
        latest = tail.iloc[-1]
        support = safe_num(latest["Support"], np.nan)
        resistance = safe_num(latest["Resistance"], np.nan)

        if not pd.isna(support):
            fig.add_hline(
                y=support, line_dash="dot",
                annotation_text=f"Support {support:.2f}",
                annotation_position="bottom right"
            )

        if not pd.isna(resistance):
            fig.add_hline(
                y=resistance, line_dash="dot",
                annotation_text=f"Resistance {resistance:.2f}",
                annotation_position="top right"
            )

    sig_df = pd.DataFrame(signals) if signals else pd.DataFrame()
    if not sig_df.empty:
        sig_df = sig_df[sig_df["Date"] >= tail["Date"].min()]
        buys = sig_df[sig_df["Type"] == "BUY"]
        sells = sig_df[sig_df["Type"] == "SELL"]

        if not buys.empty:
            fig.add_trace(go.Scatter(
                x=buys["Date"], y=buys["Price"] * 0.985,
                mode="markers+text",
                marker=dict(symbol="triangle-up", size=14),
                text=["B"] * len(buys), textposition="bottom center",
                name="BUY Signal", hovertext=buys["Reason"]
            ))

        if not sells.empty:
            fig.add_trace(go.Scatter(
                x=sells["Date"], y=sells["Price"] * 1.015,
                mode="markers+text",
                marker=dict(symbol="triangle-down", size=14),
                text=["S"] * len(sells), textposition="top center",
                name="SELL Signal", hovertext=sells["Reason"]
            ))

    fig.update_layout(
        title=f"{ticker} — Price + Signals + Support/Resistance",
        height=560,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=-0.15)
    )
    return fig

def make_momentum_chart(df, ticker):
    tail = df.tail(126)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tail["Date"], y=tail["MACD"], mode="lines", name="MACD"))
    fig.add_trace(go.Scatter(x=tail["Date"], y=tail["MACD_SIGNAL"], mode="lines", name="Signal"))
    fig.add_trace(go.Bar(x=tail["Date"], y=tail["MACD_HIST"], name="Histogram"))
    fig.update_layout(title=f"{ticker} MACD", height=240, template="plotly_dark")
    return fig

def make_rsi_stoch_chart(df, ticker):
    tail = df.tail(126)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tail["Date"], y=tail["RSI"], mode="lines", name="RSI"))
    fig.add_trace(go.Scatter(x=tail["Date"], y=tail["STOCH_K"], mode="lines", name="Stoch %K"))
    fig.add_trace(go.Scatter(x=tail["Date"], y=tail["STOCH_D"], mode="lines", name="Stoch %D"))
    fig.add_hline(y=70, line_dash="dash", annotation_text="70")
    fig.add_hline(y=30, line_dash="dash", annotation_text="30")
    fig.update_layout(title=f"{ticker} RSI + Stochastic RSI", height=240, template="plotly_dark")
    return fig

def make_obv_chart(df, ticker):
    tail = df.tail(126)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tail["Date"], y=tail["OBV"], mode="lines", name="OBV"))
    fig.add_trace(go.Scatter(x=tail["Date"], y=tail["OBV_MA20"], mode="lines", name="OBV MA20"))
    fig.update_layout(title=f"{ticker} OBV", height=240, template="plotly_dark")
    return fig

def make_volume_chart(df, ticker):
    tail = df.tail(126)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=tail["Date"], y=tail["Volume"], name="Volume"))
    fig.add_trace(go.Scatter(x=tail["Date"], y=tail["Volume_MA20"], mode="lines", name="Vol MA20"))
    fig.update_layout(title=f"{ticker} Volume", height=240, template="plotly_dark")
    return fig

def make_adx_chart(df, ticker):
    tail = df.tail(126)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tail["Date"], y=tail["ADX"], mode="lines", name="ADX"))
    fig.add_trace(go.Scatter(x=tail["Date"], y=tail["PLUS_DI"], mode="lines", name="+DI"))
    fig.add_trace(go.Scatter(x=tail["Date"], y=tail["MINUS_DI"], mode="lines", name="-DI"))
    fig.add_hline(y=25, line_dash="dash", annotation_text="25")
    fig.update_layout(title=f"{ticker} ADX Trend Strength", height=240, template="plotly_dark")
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

def analyze_stock(ticker, horizon_days, term_type, fetch_earnings=False, fetch_news=True):
    try:
        market_label, market_score, market_df = market_direction()
    except Exception as e:
        market_label = "Market unavailable"
        market_score = 0
        market_df = pd.DataFrame([{"Ticker": "Market", "Label": "Unavailable", "Score": 0, "Reason": str(e)}])

    df, err = load_price_data(ticker)
    if df.empty:
        return None, err

    df = add_indicators(df)
    if df.empty:
        return None, "Not enough data after indicators."

    mtf_df, mtf_final = multi_timeframe_confirmation(df)

    news_df = pd.DataFrame(columns=["Date", "Title", "Publisher", "Sentiment", "Score", "Link"])
    news_avg = 0
    news_overall = "Neutral"
    if fetch_news:
        try:
            news_df, news_avg, news_overall = get_news_sentiment(ticker)
        except Exception:
            news_df = pd.DataFrame(columns=["Date", "Title", "Publisher", "Sentiment", "Score", "Link"])
            news_avg = 0
            news_overall = "News unavailable"

    latest = df.iloc[-1]
    signals = detect_buysell_signals(df)

    score, risk, reasons, subscores = score_stock(
        df, latest, term_type, market_label, mtf_final, news_overall
    )

    est_df, er, label = estimate_future_price(df, horizon_days)

    adx = safe_num(latest["ADX"], 20)
    vol_ratio = safe_num(latest["Volume_Ratio"], 1.0)
    obv_trend = safe_num(latest["OBV_Trend"], 0)

    sig = final_signal(score, risk, er, term_type, mtf_final)
    conf = confidence_score(
        score, risk, er, adx, vol_ratio, obv_trend,
        market_label, mtf_final, news_overall
    )

    plan = trade_plan(latest, sig, conf, er, horizon_days)

    earnings_warning = None
    if fetch_earnings:
        ed = get_earnings_date(ticker)
        if ed:
            days_to = (ed - datetime.today().date()).days
            if 0 <= days_to <= horizon_days:
                earnings_warning = f"⚠️ Earnings in {days_to} day(s) ({ed}) — within forecast horizon."

    last_signal = signals[-1] if signals else None

    return {
        "df": df,
        "latest": latest,
        "score": score,
        "subscores": subscores,
        "risk": risk,
        "reasons": reasons,
        "signals": signals,
        "last_signal": last_signal,
        "est_df": est_df,
        "er": er,
        "label": label,
        "signal": sig,
        "confidence": conf,
        "plan": plan,
        "adx": adx,
        "vol_ratio": vol_ratio,
        "obv_trend": obv_trend,
        "bb_pct": safe_num(latest["BB_PCT"], 0.5),
        "rsi": safe_num(latest["RSI"], 50),
        "stoch_k": safe_num(latest["STOCH_K"], 50),
        "earnings_warning": earnings_warning,
        "market_label": market_label,
        "market_score": market_score,
        "market_df": market_df,
        "mtf_df": mtf_df,
        "mtf_final": mtf_final,
        "news_df": news_df,
        "news_avg": news_avg,
        "news_overall": news_overall,
        "support": safe_num(latest["Support"], np.nan),
        "resistance": safe_num(latest["Resistance"], np.nan),
    }, ""

def scan_parallel(tickers, horizon_days, term_type, max_workers=2):
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(analyze_stock, t, horizon_days, term_type, False, False): t
            for t in tickers
        }

        for future in as_completed(futures):
            t = futures[future]
            try:
                res, err = future.result(timeout=45)
                results[t] = (res, err)
            except Exception as e:
                results[t] = (None, str(e))

    return results

def build_scanner_categories(sdf):
    if sdf.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    strongest = sdf.sort_values(
        ["Score", "Confidence", "ADX"],
        ascending=[False, False, False]
    ).head(10)

    oversold = sdf[
        (sdf["RSI"] <= 45) | (sdf["Stoch K"] <= 30)
    ].sort_values(
        ["Confidence", "Score"],
        ascending=[False, False]
    ).head(10)

    breakout = sdf[
        (sdf["Vol Ratio"] >= 1.2) & (sdf["ADX"] >= 20)
    ].sort_values(
        ["Vol Ratio", "ADX", "Score"],
        ascending=[False, False, False]
    ).head(10)

    return strongest, oversold, breakout

st.sidebar.header("⚙️ Settings")

term_type = st.sidebar.radio("Trading Style", ["Short-Term", "Long-Term"])

selected_ticker = st.sidebar.selectbox(
    "Select Stock",
    ALL_TICKERS,
    index=ALL_TICKERS.index("NVDA") if "NVDA" in ALL_TICKERS else 0
)

horizon_days = st.sidebar.selectbox(
    "Forecast Horizon (days)",
    [1, 3, 5, 10, 14, 30, 60, 90],
    index=2
)

scan_count = st.sidebar.selectbox(
    "Scanner — how many stocks?",
    [5, 10, 25],
    index=0
)

fetch_earnings = st.sidebar.checkbox("Check earnings dates, slower", value=False)
fetch_news = st.sidebar.checkbox("Check news sentiment, slower", value=True)
show_sr = st.sidebar.checkbox("Show support/resistance on chart", value=True)

st.sidebar.markdown("---")
st.sidebar.info(
    "If Yahoo shows Too Many Requests, wait 10-30 minutes, keep scanner at 5, "
    "and avoid refreshing repeatedly."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Single Stock",
    "🔍 Scanner",
    "🌎 Market Dashboard",
    "📋 Ticker List"
])

with tab1:
    st.subheader(f"Analysis — {selected_ticker}")

    with st.spinner(f"Loading {selected_ticker}…"):
        result, error = analyze_stock(
            selected_ticker,
            horizon_days,
            term_type,
            fetch_earnings,
            fetch_news
        )

    if result is None:
        st.error(f"Could not analyse {selected_ticker}: {error}")
        st.info("Try again later, or change the ticker. Yahoo Finance may be temporarily rate-limiting your app.")
    else:
        latest = result["latest"]

        if result["earnings_warning"]:
            st.warning(result["earnings_warning"])

        c1, c2, c3 = st.columns(3)
        c1.metric("Market Direction", result["market_label"], f"Score {result['market_score']}")
        c2.metric("Multi-Timeframe", result["mtf_final"])
        c3.metric("News Sentiment", result["news_overall"], f"{result['news_avg']:.2f}")

        st.markdown("### 🌎 Market Context")
        st.dataframe(result["market_df"], use_container_width=True)

        st.markdown("### 🧭 Multi-Timeframe Confirmation")
        st.dataframe(result["mtf_df"], use_container_width=True)

        ls = result["last_signal"]
        if ls:
            if ls["Type"] == "BUY":
                st.success(
                    f"🟢 Last auto-signal: BUY on "
                    f"{pd.Timestamp(ls['Date']).strftime('%b %d %Y')} "
                    f"@ ${ls['Price']:.2f} — {ls['Reason']}"
                )
            else:
                st.error(
                    f"🔴 Last auto-signal: SELL on "
                    f"{pd.Timestamp(ls['Date']).strftime('%b %d %Y')} "
                    f"@ ${ls['Price']:.2f} — {ls['Reason']}"
                )

        cols = st.columns(10)
        metrics = [
            ("Price", f"${safe_num(latest['Close']):.2f}"),
            ("Signal", result["signal"][:20]),
            ("Score", f"{result['score']:.1f}/10"),
            ("Confidence", f"{result['confidence']}%"),
            ("Risk", result["risk"]),
            ("ADX", f"{result['adx']:.1f}"),
            ("RSI", f"{result['rsi']:.1f}"),
            ("Stoch K", f"{result['stoch_k']:.1f}"),
            ("Support", f"${result['support']:.2f}" if not pd.isna(result["support"]) else "—"),
            ("Resistance", f"${result['resistance']:.2f}" if not pd.isna(result["resistance"]) else "—"),
        ]

        for col, (label, val) in zip(cols, metrics):
            col.metric(label, val)

        signal_box(result["signal"])

        st.markdown("### 📋 Trade Plan")
        st.dataframe(result["plan"], use_container_width=True)

        st.markdown("### 📈 Price Chart + Buy / Sell Signals + Support / Resistance")
        st.plotly_chart(
            make_price_chart(result["df"], selected_ticker, result["signals"], show_sr=show_sr),
            use_container_width=True
        )

        if result["signals"]:
            st.markdown("### 🔔 Historical Buy/Sell Signals")
            sig_df = pd.DataFrame(result["signals"])
            sig_df["Date"] = pd.to_datetime(sig_df["Date"]).dt.strftime("%Y-%m-%d")
            sig_df["Price"] = sig_df["Price"].round(2)
            st.dataframe(sig_df, use_container_width=True, height=280)

        r1c1, r1c2 = st.columns(2)
        with r1c1:
            st.plotly_chart(make_momentum_chart(result["df"], selected_ticker), use_container_width=True)
        with r1c2:
            st.plotly_chart(make_rsi_stoch_chart(result["df"], selected_ticker), use_container_width=True)

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            st.plotly_chart(make_obv_chart(result["df"], selected_ticker), use_container_width=True)
        with r2c2:
            st.plotly_chart(make_volume_chart(result["df"], selected_ticker), use_container_width=True)

        st.plotly_chart(make_adx_chart(result["df"], selected_ticker), use_container_width=True)

        st.markdown("### 🔮 Price Forecast")
        st.dataframe(result["est_df"], use_container_width=True)

        st.markdown("### 🧮 Score Breakdown")
        sb = result["subscores"]
        score_rows = [{"Factor": k, "Score /10": v} for k, v in sb.items()]
        st.dataframe(pd.DataFrame(score_rows), use_container_width=True)

        st.markdown("### 📰 Recent News Sentiment")
        if result["news_df"].empty:
            st.info("No recent Yahoo news found for this ticker.")
        else:
            news_cols = ["Date", "Title", "Publisher", "Sentiment", "Score", "Link"]
            news_show = result["news_df"].copy()

            for col in news_cols:
                if col not in news_show.columns:
                    news_show[col] = ""

            news_show = news_show[news_cols]
            news_show = news_show[news_show["Title"].astype(str).str.strip() != ""]

            if news_show.empty:
                st.info("No recent Yahoo news found for this ticker.")
            else:
                st.dataframe(news_show, use_container_width=True, height=300)

        st.markdown("### 💬 Signal Reasoning")
        for r in result["reasons"]:
            st.write(f"- {r}")

with tab2:
    st.subheader("⚡ Scanner + Ranking Categories")
    st.caption("This scanner runs slowly to reduce Yahoo rate-limit issues.")

    scan_tickers = ALL_TICKERS[:scan_count]

    if st.button("▶️ Run Scanner", type="primary"):
        progress_bar = st.progress(0, text="Starting scan…")

        with st.spinner("Scanning…"):
            raw_results = scan_parallel(
                scan_tickers,
                horizon_days,
                term_type,
                max_workers=2
            )

        progress_bar.progress(1.0, text="Done!")

        rows = []
        failed = []

        for ticker in scan_tickers:
            res, err = raw_results.get(ticker, (None, "timeout"))
            if res is None:
                failed.append({"Ticker": ticker, "Error": err})
                continue

            ls = res["last_signal"]
            rows.append({
                "Ticker": ticker,
                "Price": round(safe_num(res["latest"]["Close"]), 2),
                "Signal": res["signal"],
                "Score": res["score"],
                "Confidence": res["confidence"],
                "Est. Return %": round(res["er"] * 100, 2),
                "Risk": res["risk"],
                "ADX": round(res["adx"], 1),
                "RSI": round(res["rsi"], 1),
                "Stoch K": round(res["stoch_k"], 1),
                "OBV Trend": "↑ Bullish" if res["obv_trend"] > 0 else "↓ Bearish",
                "Vol Ratio": round(res["vol_ratio"], 2),
                "BB%": round(res["bb_pct"], 2),
                "Market": res["market_label"],
                "MTF": res["mtf_final"],
                "Last Signal": f"{ls['Type']} @ ${ls['Price']:.2f}" if ls else "—",
            })

        if rows:
            order = {
                "🔥 Strong Buy": 1,
                "🚀 Strong Long-Term Buy": 1,
                "✅ Buy Signal": 2,
                "✅ Long-Term Buy": 2,
                "📉 Buy on Dip": 3,
                "📉 Long-Term Buy on Dip": 3,
                "⏳ Hold / Wait": 4,
                "⏳ Long-Term Hold / Watch": 4,
                "⚠️ Avoid / High Risk": 5,
                "🔻 Sell / High Caution": 6,
                "⚠️ Avoid Long-Term": 7,
            }

            sdf = pd.DataFrame(rows)
            sdf["_ord"] = sdf["Signal"].map(order).fillna(9)
            sdf = sdf.sort_values(
                ["_ord", "Confidence", "Score"],
                ascending=[True, False, False]
            ).drop(columns=["_ord"])

            strongest, oversold, breakout = build_scanner_categories(sdf)

            st.success(f"✅ Scanned {len(sdf)} stocks")

            st.markdown("### 🏆 Top Strongest Momentum")
            st.dataframe(strongest, use_container_width=True, height=300)

            st.markdown("### 🟡 Top Oversold Bounce Setups")
            st.dataframe(oversold, use_container_width=True, height=300)

            st.markdown("### 🚀 Top Breakout Candidates")
            st.dataframe(breakout, use_container_width=True, height=300)

            st.markdown("### 📋 Full Scanner Results")
            st.dataframe(sdf, use_container_width=True, height=500)

            st.download_button(
                "⬇️ Download Results CSV",
                sdf.to_csv(index=False).encode("utf-8"),
                "scanner_v4.csv",
                "text/csv"
            )

        if failed:
            with st.expander("Show failed tickers"):
                st.dataframe(pd.DataFrame(failed), use_container_width=True)

        if not rows:
            st.warning("No results returned. Yahoo may be rate-limiting your app. Wait 10-30 minutes and try again.")
    else:
        st.info("Press Run Scanner to start. Keep scanner count at 5 if you are getting rate-limit errors.")

with tab3:
    st.subheader("🌎 Market Dashboard")

    with st.spinner("Loading market direction…"):
        try:
            ml, ms, mdf = market_direction()
        except Exception as e:
            ml, ms = "Market unavailable", 0
            mdf = pd.DataFrame([{"Ticker": "Market", "Label": "Unavailable", "Score": 0, "Reason": str(e)}])

    st.metric("Overall Market Direction", ml, f"Score {ms}")
    st.dataframe(mdf, use_container_width=True)

    st.markdown("""
    **How to read this:**

    - SPY = broad market
    - QQQ = tech / growth market
    - VIX = fear index

    If SPY and QQQ are bullish and VIX is low/normal, buy signals are more reliable.
    If market is risk-off, even strong stocks can pull back.
    """)

with tab4:
    st.subheader("📋 Ticker List")
    st.write(f"Total tickers loaded: **{len(ALL_TICKERS)}**")
    st.dataframe(pd.DataFrame({"Ticker": ALL_TICKERS}), use_container_width=True, height=700)
