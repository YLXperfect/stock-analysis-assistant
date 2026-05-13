"""技术分析模块

基于 K 线数据计算技术指标，提供纯函数式分析。
不依赖 MCP，接收 pandas DataFrame 输入。
"""

import pandas as pd
import numpy as np
from typing import Sequence, TypedDict


class TechResult(TypedDict):
    """技术分析结果"""
    trend: str           # 多头 / 空头 / 震荡
    ma_signal: str       # 均线信号描述
    macd_signal: str     # MACD 信号描述
    rsi_value: float     # RSI 数值
    rsi_signal: str      # RSI 信号描述
    bollinger_pos: str   # 布林带位置
    support: float       # 支撑位
    resistance: float    # 压力位
    score: int           # 技术面评分 1-10
    summary: str         # 一句话总结


def calc_ma(df: pd.DataFrame, periods: Sequence[int] = (5, 10, 20, 60)) -> pd.DataFrame:
    """计算移动平均线"""
    for p in periods:
        df[f"ma{p}"] = df["close"].rolling(p).mean()
    return df


def calc_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """计算 MACD"""
    ema_fast = df["close"].ewm(span=fast).mean()
    ema_slow = df["close"].ewm(span=slow).mean()
    df["dif"] = ema_fast - ema_slow
    df["dea"] = df["dif"].ewm(span=signal).mean()
    df["macd_hist"] = 2 * (df["dif"] - df["dea"])
    return df


def calc_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """计算 RSI"""
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    avg_loss = pd.Series(avg_loss).replace(0, 1e-10)
    rs = avg_gain / avg_loss
    df[f"rsi{period}"] = 100 - (100 / (1 + rs))
    return df


def calc_bollinger(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    """计算布林带"""
    mid = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    df["boll_mid"] = mid
    df["boll_upper"] = mid + std_dev * std
    df["boll_lower"] = mid - std_dev * std
    return df


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """计算 Average True Range (ATR)"""
    high: "pd.Series" = pd.to_numeric(df["high"], errors="coerce")  # type: ignore[assignment]
    low: "pd.Series" = pd.to_numeric(df["low"], errors="coerce")  # type: ignore[assignment]
    close_s: "pd.Series" = pd.to_numeric(df["close"], errors="coerce")  # type: ignore[assignment]
    prev_close = close_s.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df[f"atr{period}"] = tr.rolling(period).mean()
    return df


def enrich_candlesticks(df: pd.DataFrame) -> pd.DataFrame:
    """一次性计算所有技术指标"""
    df = calc_ma(df)
    df = calc_macd(df)
    df = calc_rsi(df)
    df = calc_bollinger(df)
    df = calc_atr(df)
    return df


def analyze_technicals(df: pd.DataFrame) -> TechResult:
    """综合技术分析

    Args:
        df: 至少包含 close 列的 K 线 DataFrame，建议 60+ 根

    Returns:
        技术分析结果字典
    """
    df = enrich_candlesticks(df.copy())

    last = df.iloc[-1]
    close = float(last["close"])
    score = 5  # 基准分

    # --- 均线趋势 ---
    ma5 = last.get("ma5")
    ma10 = last.get("ma10")
    ma20 = last.get("ma20")
    ma60 = last.get("ma60")

    if all(v is not None and not pd.isna(v) for v in [ma5, ma10, ma20, ma60]):
        if close > ma5 > ma10 > ma20 > ma60:
            trend = "多头排列"
            score += 2
        elif close < ma5 < ma10 < ma20 < ma60:
            trend = "空头排列"
            score -= 2
        else:
            trend = "震荡整理"

        if close > ma20:
            ma_signal = f"价格在 MA20({ma20:.2f}) 上方"
            score += 1
        else:
            ma_signal = f"价格在 MA20({ma20:.2f}) 下方"
            score -= 1
    else:
        trend = "数据不足"
        ma_signal = "均线数据不完整"

    # --- MACD ---
    dif = last.get("dif", 0)
    dea = last.get("dea", 0)
    hist = last.get("macd_hist", 0)

    if pd.isna(dif):
        macd_signal = "MACD 数据不足"
    elif dif > dea and hist > 0:
        macd_signal = f"DIF({dif:.3f}) > DEA，金叉/多头"
        score += 1
    elif dif < dea and hist < 0:
        macd_signal = f"DIF({dif:.3f}) < DEA，死叉/空头"
        score -= 1
    else:
        macd_signal = f"DIF({dif:.3f}) DEA 趋势转换中"

    # --- RSI ---
    rsi_val = last.get("rsi14", 50)
    if pd.isna(rsi_val):
        rsi_val = 50.0
    rsi_val = float(rsi_val)

    if rsi_val > 70:
        rsi_signal = "超买区间"
        score -= 1
    elif rsi_val > 60:
        rsi_signal = "偏强"
    elif rsi_val < 30:
        rsi_signal = "超卖区间"
        score += 1
    elif rsi_val < 40:
        rsi_signal = "偏弱"
    else:
        rsi_signal = "中性"

    # --- 布林带 ---
    boll_upper = last.get("boll_upper")
    boll_lower = last.get("boll_lower")
    if boll_upper and boll_lower and not pd.isna(boll_upper):
        if close > boll_upper:
            bollinger_pos = "突破上轨"
            score -= 1
        elif close < boll_lower:
            bollinger_pos = "跌破下轨"
            score += 1
        else:
            bollinger_pos = f"通道内 ({boll_lower:.2f} - {boll_upper:.2f})"
    else:
        bollinger_pos = "数据不足"

    # --- 支撑/压力 ---
    recent = df.tail(20)
    low_vals: "pd.Series" = pd.to_numeric(recent["low"], errors="coerce")  # type: ignore[assignment]
    high_vals: "pd.Series" = pd.to_numeric(recent["high"], errors="coerce")  # type: ignore[assignment]
    support = float(low_vals.min())  # type: ignore[union-attr]
    resistance = float(high_vals.max())  # type: ignore[union-attr]

    # --- 评分归一化 ---
    score = max(1, min(10, score))

    summary = f"趋势:{trend} | RSI:{rsi_val:.0f}({rsi_signal}) | 支撑:{support:.2f} 压力:{resistance:.2f}"

    return TechResult(
        trend=trend,
        ma_signal=ma_signal,
        macd_signal=macd_signal,
        rsi_value=round(rsi_val, 1),
        rsi_signal=rsi_signal,
        bollinger_pos=bollinger_pos,
        support=round(support, 3),
        resistance=round(resistance, 3),
        score=score,
        summary=summary,
    )
