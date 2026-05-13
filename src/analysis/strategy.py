"""交易策略分析模块

基于技术指标和用户风险偏好，生成入场/离场/定投策略建议。
含多空辩论机制（借鉴 TradingAgents）。
"""

from typing import Any
import pandas as pd
import numpy as np
from typing import TypedDict, Optional


class TradeProposal(TypedDict):
    action: str              # buy / hold / sell / wait
    confidence: str          # high / medium / low
    entry_price: str         # 入场价区间
    stop_loss: str           # 止损价
    take_profit: str         # 止盈价
    risk_reward_ratio: str   # 风险收益比
    position_advice: str     # 仓位建议
    bull_case: str           # 看多论据
    bear_case: str           # 看空论据
    reasoning: str           # 综合理由


def calc_stop_loss(
    entry_price: float,
    atr: float = 0,
    method: str = "atr",
    risk_pct: float = 8.0,
    support_level: float = 0,
) -> float:
    """计算止损价

    Args:
        entry_price: 入场价
        atr: ATR 值（atr 方法时需要）
        method: atr / fixed / support
        risk_pct: 固定百分比方法的止损比例
        support_level: 支撑位（support 方法时需要）
    """
    if method == "atr" and atr > 0:
        return round(entry_price - 2 * atr, 2)
    elif method == "support" and support_level > 0:
        return round(support_level * 0.98, 2)  # 支撑位下方 2%
    else:
        return round(entry_price * (1 - risk_pct / 100), 2)


def calc_take_profit(
    entry_price: float,
    stop_loss: float,
    reward_ratio: float = 2.0,
) -> float:
    """计算止盈价（基于风险收益比）"""
    risk = entry_price - stop_loss
    return round(entry_price + risk * reward_ratio, 2)


def calc_risk_reward(entry: float, stop_loss: float, take_profit: float) -> str:
    """计算风险收益比"""
    risk = abs(entry - stop_loss)
    reward = abs(take_profit - entry)
    if risk <= 0:
        return "N/A"
    ratio = reward / risk
    return f"1:{ratio:.1f}"


def _assess_technical_bias(tech_result: dict) -> str:
    """从技术分析结果判断偏向"""
    score = tech_result.get("score", 5)
    trend = tech_result.get("trend", "")
    if score >= 7 and "多头" in trend:
        return "bullish"
    elif score <= 3 and "空头" in trend:
        return "bearish"
    else:
        return "neutral"


def generate_bull_bear_debate(
    tech_result: "dict[str, Any]",
    fundamental_score: int = 5,
    capital_flow_json: str = "",
    news_summary: str = "",
) -> "dict[str, Any]":
    """生成多空辩论论据

    基于技术面、基本面、资金面、新闻面生成看多和看空论据。
    """
    bull_points = []
    bear_points = []

    # 技术面论据
    tech_score = tech_result.get("score", 5)
    trend = tech_result.get("trend", "")
    rsi = tech_result.get("rsi_value", 50)
    macd_signal = tech_result.get("macd_signal", "")

    if tech_score >= 7:
        bull_points.append(f"技术面评分 {tech_score}/10，趋势向好")
    elif tech_score <= 3:
        bear_points.append(f"技术面评分 {tech_score}/10，趋势偏弱")

    if "多头" in trend:
        bull_points.append(f"均线呈多头排列")
    elif "空头" in trend:
        bear_points.append(f"均线呈空头排列")

    if "金叉" in macd_signal:
        bull_points.append("MACD 金叉信号")
    elif "死叉" in macd_signal:
        bear_points.append("MACD 死叉信号")

    if rsi < 30:
        bull_points.append(f"RSI {rsi:.0f} 超卖，反弹概率大")
    elif rsi > 70:
        bear_points.append(f"RSI {rsi:.0f} 超买，回调风险高")
    elif 40 <= rsi <= 60:
        bull_points.append(f"RSI {rsi:.0f} 中性区间，方向待确认")

    support = tech_result.get("support", 0)
    resistance = tech_result.get("resistance", 0)
    if support > 0:
        bull_points.append(f"支撑位 {support:.2f}")
    if resistance > 0:
        bear_points.append(f"压力位 {resistance:.2f}")

    # 基本面论据
    if fundamental_score >= 7:
        bull_points.append(f"基本面评分 {fundamental_score}/10，质地优秀")
    elif fundamental_score <= 4:
        bear_points.append(f"基本面评分 {fundamental_score}/10，存在隐忧")

    # 资金面论据
    import json
    if capital_flow_json:
        try:
            data = json.loads(capital_flow_json)
            if isinstance(data, dict):
                inflow = data.get("main_inflow", data.get("inflow", data.get("主力流入")))
                outflow = data.get("main_outflow", data.get("outflow", data.get("主力流出")))
                if inflow and outflow:
                    net = _safe_float(inflow) - _safe_float(outflow)
                    if net > 0:
                        bull_points.append(f"主力资金净流入")
                    else:
                        bear_points.append(f"主力资金净流出")
        except Exception:
            pass

    # 新闻论据
    if news_summary:
        positive_kw = ["利好", "增长", "突破", "创新高", "超预期", "获批", "合作"]
        negative_kw = ["利空", "下跌", "暴雷", "亏损", "风险", "处罚", "减持", "召回"]
        for kw in positive_kw:
            if kw in news_summary:
                bull_points.append(f"消息面有利好关键词: {kw}")
                break
        for kw in negative_kw:
            if kw in news_summary:
                bear_points.append(f"消息面有利空关键词: {kw}")
                break

    # 综合评判
    bull_count = len(bull_points)
    bear_count = len(bear_points)
    if bull_count > bear_count + 1:
        verdict = "偏多"
        confidence = "high" if bull_count > bear_count + 2 else "medium"
    elif bear_count > bull_count + 1:
        verdict = "偏空"
        confidence = "high" if bear_count > bull_count + 2 else "medium"
    else:
        verdict = "中性"
        confidence = "medium"

    return {
        "bull_case": "；".join(bull_points) if bull_points else "无明显看多信号",
        "bear_case": "；".join(bear_points) if bear_points else "无明显看空信号",
        "verdict": verdict,
        "confidence": confidence,
        "bull_count": bull_count,
        "bear_count": bear_count,
    }


def analyze_entry_strategy(
    current_price: float,
    support: float,
    resistance: float,
    atr: float,
    tech_score: int,
    risk_tolerance: str = "moderate",
    fundamental_score: int = 5,
) -> TradeProposal:
    """入场时机分析"""
    # 风险偏好映射
    risk_config = {
        "conservative": {"sl_pct": 5, "reward_ratio": 2.0, "max_pos": 10},
        "moderate": {"sl_pct": 8, "reward_ratio": 2.0, "max_pos": 20},
        "aggressive": {"sl_pct": 12, "reward_ratio": 1.5, "max_pos": 30},
    }
    cfg = risk_config.get(risk_tolerance, risk_config["moderate"])

    # 止损计算
    stop_loss = calc_stop_loss(current_price, atr=atr, method="atr")
    # 安全检查：止损不能低于支撑位
    if support > 0 and stop_loss < support * 0.95:
        stop_loss = round(support * 0.98, 2)

    take_profit = calc_take_profit(current_price, stop_loss, cfg["reward_ratio"])
    rr = calc_risk_reward(current_price, stop_loss, take_profit)

    # 入场价区间
    if support > 0 and current_price > support:
        entry_low = round(support * 1.01, 2)
        entry_high = round(support * 1.03, 2)
        entry_price = f"{entry_low}-{entry_high}（支撑位附近）"
    else:
        entry_price = f"{current_price:.2f}（当前价）"

    # 仓位建议
    pos_pct = cfg["max_pos"]
    if tech_score >= 7:
        position_advice = f"总资金的 {pos_pct}%（技术面强势）"
    elif tech_score >= 5:
        position_advice = f"总资金的 {pos_pct // 2}%（技术面中性，半仓）"
    else:
        position_advice = f"总资金的 {pos_pct // 3}%（技术面偏弱，轻仓试探）"

    # 行动决策
    if tech_score >= 7 and fundamental_score >= 6:
        action = "buy"
        confidence = "high"
    elif tech_score >= 5 and fundamental_score >= 5:
        action = "buy"
        confidence = "medium"
    elif tech_score <= 3:
        action = "wait"
        confidence = "high"
    else:
        action = "wait"
        confidence = "medium"

    sl_pct = ((current_price - stop_loss) / current_price * 100) if current_price > 0 else 0
    tp_pct = ((take_profit - current_price) / current_price * 100) if current_price > 0 else 0

    reasoning = f"技术面 {tech_score}/10，基本面 {fundamental_score}/10。"
    if action == "buy":
        reasoning += f"建议在支撑位附近入场，止损 {stop_loss}(-{sl_pct:.1f}%)，止盈 {take_profit}(+{tp_pct:.1f}%)。"
    else:
        reasoning += "当前不宜追高，建议等待更好的入场时机。"

    return TradeProposal(
        action=action,
        confidence=confidence,
        entry_price=entry_price,
        stop_loss=f"{stop_loss} (-{sl_pct:.1f}%)",
        take_profit=f"{take_profit} (+{tp_pct:.1f}%)",
        risk_reward_ratio=rr,
        position_advice=position_advice,
        bull_case="",
        bear_case="",
        reasoning=reasoning,
    )


def analyze_dca_strategy(
    current_price: float,
    support: float,
    resistance: float,
    total_budget: float = 10000,
    intervals: int = 3,
) -> dict:
    """定投策略分析（Dollar Cost Averaging）"""
    # 在支撑位到当前价之间均匀分批
    low = support if support > 0 else current_price * 0.95
    high = current_price
    step = (high - low) / intervals if intervals > 1 else 0

    batches = []
    per_budget = total_budget / intervals
    for i in range(intervals):
        price = round(low + step * i, 2) if step > 0 else current_price
        shares = int(per_budget / price) if price > 0 else 0
        batches.append({
            "batch": i + 1,
            "price": f"{price:.2f}",
            "budget": f"{per_budget:.0f}",
            "shares": shares,
        })

    avg_price = round(sum(float(b["price"]) for b in batches) / len(batches), 2)
    total_shares = sum(b["shares"] for b in batches)

    return {
        "strategy": "DCA 定投",
        "total_budget": total_budget,
        "intervals": intervals,
        "batches": batches,
        "avg_entry_price": f"{avg_price:.2f}",
        "total_shares": total_shares,
        "current_price": f"{current_price:.2f}",
        "discount": f"{((current_price - avg_price) / current_price * 100):.1f}%" if current_price > 0 else "N/A",
    }


def _safe_float(val, default=0.0) -> float:
    if val is None:
        return default
    try:
        return float(str(val).replace(",", "").replace("%", ""))
    except (ValueError, TypeError):
        return default


def format_trade_proposal(proposal: TradeProposal, debate: dict, symbol: str) -> str:
    """格式化交易提案为 Markdown"""
    action_cn = {"buy": "建议买入", "hold": "建议持有", "sell": "建议卖出", "wait": "建议观望"}
    lines = [
        f"## 交易策略: {symbol}",
        f"",
        f"**{action_cn.get(proposal['action'], proposal['action'])}** | 信心度: {proposal['confidence']}",
        f"",
        f"| 项目 | 建议 |",
        f"|------|------|",
        f"| 入场价 | {proposal['entry_price']} |",
        f"| 止损价 | {proposal['stop_loss']} |",
        f"| 止盈价 | {proposal['take_profit']} |",
        f"| 风险收益比 | {proposal['risk_reward_ratio']} |",
        f"| 仓位建议 | {proposal['position_advice']} |",
        f"",
        f"### 多空辩论",
        f"",
        f"**看多:** {debate.get('bull_case', 'N/A')}",
        f"",
        f"**看空:** {debate.get('bear_case', 'N/A')}",
        f"",
        f"**综合判断:** {debate.get('verdict', 'N/A')} (信心: {debate.get('confidence', 'N/A')})",
        f"",
        f"> {proposal['reasoning']}",
    ]
    return "\n".join(lines)


def format_dca_report(result: dict, symbol: str) -> str:
    """格式化定投分析为 Markdown"""
    lines = [
        f"## 定投策略: {symbol}",
        f"",
        f"总预算: {result['total_budget']} | 分 {result['intervals']} 批",
        f"",
        f"| 批次 | 价格 | 金额 | 股数 |",
        f"|------|------|------|------|",
    ]
    for b in result["batches"]:
        lines.append(f"| 第{b['batch']}批 | {b['price']} | {b['budget']} | {b['shares']} |")
    lines.extend([
        f"",
        f"平均成本: {result['avg_entry_price']} | 总股数: {result['total_shares']}",
        f"相比当前价 {result['current_price']} 折扣: {result['discount']}",
    ])
    return "\n".join(lines)
