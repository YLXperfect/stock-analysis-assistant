"""持仓安全性分析模块

评估单个持仓和整体组合的安全性，给出仓位调整建议。
"""

from typing import TypedDict


class PositionSafetyResult(TypedDict):
    symbol: str
    position_value: float          # 持仓市值
    portfolio_pct: float           # 占组合百分比
    safety_level: str              # safe / caution / danger
    unrealized_pnl_pct: float      # 浮盈/亏百分比
    suggested_stop_loss: float     # 建议止损价
    suggested_action: str          # hold / reduce / add / exit
    risk_factors: list[str]        # 风险因素
    score: int                     # 安全评分 1-10


def evaluate_position_safety(
    symbol: str,
    quantity: float,
    cost_price: float,
    current_price: float,
    portfolio_total: float,
    tech_score: int = 5,
    tech_trend: str = "",
    atr: float = 0,
    risk_tolerance: str = "moderate",
    max_position_pct: float = 20.0,
    stop_loss_default_pct: float = 8.0,
) -> PositionSafetyResult:
    """评估单个持仓的安全性

    Args:
        symbol: 股票代码
        quantity: 持仓数量
        cost_price: 成本价
        current_price: 当前价
        portfolio_total: 组合总资产
        tech_score: 技术面评分 1-10
        tech_trend: 技术面趋势描述
        atr: ATR 值
        risk_tolerance: 风险偏好
        max_position_pct: 单只最大仓位占比
        stop_loss_default_pct: 默认止损比例
    """
    position_value = quantity * current_price
    portfolio_pct = (position_value / portfolio_total * 100) if portfolio_total > 0 else 0
    unrealized_pnl_pct = ((current_price - cost_price) / cost_price * 100) if cost_price > 0 else 0

    risk_factors = []
    score = 10

    # 1. 仓位集中度
    if portfolio_pct > max_position_pct * 1.5:
        risk_factors.append(f"仓位过重: 占组合 {portfolio_pct:.1f}%（上限 {max_position_pct}%）")
        score -= 3
    elif portfolio_pct > max_position_pct:
        risk_factors.append(f"仓位偏重: 占组合 {portfolio_pct:.1f}%（上限 {max_position_pct}%）")
        score -= 1

    # 2. 技术面趋势
    if "空头" in tech_trend:
        risk_factors.append(f"技术面空头排列（评分 {tech_score}/10）")
        score -= 2
    elif "震荡" in tech_trend and tech_score <= 4:
        risk_factors.append(f"技术面震荡偏弱（评分 {tech_score}/10）")
        score -= 1

    # 3. 浮亏程度
    if unrealized_pnl_pct < -15:
        risk_factors.append(f"深度浮亏 {unrealized_pnl_pct:.1f}%")
        score -= 2
    elif unrealized_pnl_pct < -8:
        risk_factors.append(f"浮亏 {unrealized_pnl_pct:.1f}%")
        score -= 1

    # 4. 持仓逆势
    if unrealized_pnl_pct < -5 and tech_score >= 6:
        risk_factors.append("持仓逆势下跌（技术面尚可但价格下跌）")

    # 计算建议止损价
    if atr > 0:
        suggested_sl = round(current_price - 2 * atr, 2)
    else:
        suggested_sl = round(current_price * (1 - stop_loss_default_pct / 100), 2)

    # 安全等级
    score = max(1, min(10, score))
    if score >= 7:
        safety_level = "safe"
    elif score >= 4:
        safety_level = "caution"
    else:
        safety_level = "danger"

    # 建议操作
    if safety_level == "danger":
        suggested_action = "reduce" if unrealized_pnl_pct < -10 else "exit"
    elif safety_level == "caution":
        if unrealized_pnl_pct > 10:
            suggested_action = "hold"  # 有浮盈，谨慎持有
        else:
            suggested_action = "reduce"  # 适当减仓
    else:
        if unrealized_pnl_pct < -3 and tech_score <= 4:
            suggested_action = "hold"
        else:
            suggested_action = "add" if tech_score >= 6 else "hold"

    return PositionSafetyResult(
        symbol=symbol,
        position_value=round(position_value, 2),
        portfolio_pct=round(portfolio_pct, 1),
        safety_level=safety_level,
        unrealized_pnl_pct=round(unrealized_pnl_pct, 1),
        suggested_stop_loss=suggested_sl,
        suggested_action=suggested_action,
        risk_factors=risk_factors if risk_factors else ["无明显风险"],
        score=score,
    )


def evaluate_portfolio_safety(
    positions: list[dict],
    portfolio_total: float,
    risk_tolerance: str = "moderate",
    max_position_pct: float = 20.0,
) -> list[PositionSafetyResult]:
    """评估整个投资组合的持仓安全性

    Args:
        positions: [{symbol, quantity, cost_price, current_price, tech_score, tech_trend, atr}]
        portfolio_total: 组合总资产
        risk_tolerance: 风险偏好
        max_position_pct: 单只最大仓位占比
    """
    results = []
    for pos in positions:
        result = evaluate_position_safety(
            symbol=pos.get("symbol", "?"),
            quantity=pos.get("quantity", 0),
            cost_price=pos.get("cost_price", 0),
            current_price=pos.get("current_price", 0),
            portfolio_total=portfolio_total,
            tech_score=pos.get("tech_score", 5),
            tech_trend=pos.get("tech_trend", ""),
            atr=pos.get("atr", 0),
            risk_tolerance=risk_tolerance,
            max_position_pct=max_position_pct,
        )
        results.append(result)

    # 按安全评分排序（最危险的在前）
    results.sort(key=lambda x: x["score"])
    return results


def format_position_report(result: PositionSafetyResult) -> str:
    """格式化单个持仓安全报告"""
    level_cn = {"safe": "安全", "caution": "注意", "danger": "危险"}
    action_cn = {"hold": "持有", "reduce": "减仓", "add": "加仓", "exit": "清仓"}

    lines = [
        f"### {result['symbol']}",
        f"",
        f"| 项目 | 评估 |",
        f"|------|------|",
        f"| 安全等级 | **{level_cn.get(result['safety_level'], result['safety_level'])}** ({result['score']}/10) |",
        f"| 持仓市值 | {result['position_value']:.2f} |",
        f"| 占组合 | {result['portfolio_pct']:.1f}% |",
        f"| 浮盈亏 | {result['unrealized_pnl_pct']:+.1f}% |",
        f"| 建议止损 | {result['suggested_stop_loss']} |",
        f"| 建议操作 | **{action_cn.get(result['suggested_action'], result['suggested_action'])}** |",
        f"",
        f"风险因素: {'; '.join(result['risk_factors'])}",
    ]
    return "\n".join(lines)


def format_portfolio_report(results: list[PositionSafetyResult]) -> str:
    """格式化组合安全报告"""
    if not results:
        return "当前无持仓"

    safe_count = sum(1 for r in results if r["safety_level"] == "safe")
    caution_count = sum(1 for r in results if r["safety_level"] == "caution")
    danger_count = sum(1 for r in results if r["safety_level"] == "danger")

    lines = [
        "## 投资组合安全评估",
        "",
        f"持仓数: {len(results)} | 安全: {safe_count} | 注意: {caution_count} | 危险: {danger_count}",
        "",
    ]

    for r in results:
        lines.append(format_position_report(r))
        lines.append("")

    # 总结
    if danger_count > 0:
        lines.append("**需要关注:** " + ", ".join(r["symbol"] for r in results if r["safety_level"] == "danger"))

    return "\n".join(lines)
