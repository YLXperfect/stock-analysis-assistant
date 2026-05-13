"""策略回测模块

PineScript V6 策略模板 + 结果格式化。
Longbridge quant run 命令在服务端执行 PineScript，返回回测指标。
"""

import json

# ── 策略模板 ────────────────────────────────────────────────────

STRATEGY_TEMPLATES = {
    "ma_cross": {
        "name": "均线交叉",
        "script": """strategy("MA Cross", overlay=true)
fast = ta.sma(close, {fast_period})
slow = ta.sma(close, {slow_period})
if ta.crossover(fast, slow)
    strategy.entry("Long", strategy.long)
if ta.crossunder(fast, slow)
    strategy.close("Long")""",
    },
    "rsi": {
        "name": "RSI 超买超卖",
        "script": """strategy("RSI", overlay=false)
rsi = ta.rsi(close, {rsi_period})
if rsi < {oversold}
    strategy.entry("Long", strategy.long)
if rsi > {overbought}
    strategy.close("Long")""",
    },
    "bollinger": {
        "name": "布林带突破",
        "script": """strategy("Bollinger", overlay=true)
[middle, upper, lower] = ta.bb(close, {boll_period}, {boll_stddev})
if close < lower
    strategy.entry("Long", strategy.long)
if close > upper
    strategy.close("Long")""",
    },
    "macd": {
        "name": "MACD 金叉死叉",
        "script": """strategy("MACD")
[macd, signal, hist] = ta.macd(close, 12, 26, 9)
if ta.crossover(macd, signal)
    strategy.entry("Long", strategy.long)
if ta.crossunder(macd, signal)
    strategy.close("Long")""",
    },
}


def generate_pine_script(strategy_type: str, **params) -> str:
    """从模板生成 PineScript

    Args:
        strategy_type: 策略类型 ma_cross/rsi/bollinger/macd
        **params: 模板参数

    Returns:
        PineScript 字符串

    Raises:
        ValueError: 不支持的策略类型
    """
    template = STRATEGY_TEMPLATES.get(strategy_type)
    if not template:
        available = ", ".join(STRATEGY_TEMPLATES.keys())
        raise ValueError(
            f"不支持的策略类型: {strategy_type}。可用策略: {available}"
        )

    # 设置默认参数
    defaults = {
        "fast_period": 5, "slow_period": 20,
        "rsi_period": 14, "oversold": 30.0, "overbought": 70.0,
        "boll_period": 20, "boll_stddev": 2.0,
    }
    for key, default_val in defaults.items():
        if key not in params:
            params[key] = default_val

    return template["script"].format(**params)


def format_backtest_result(raw_output: str, symbol: str, strategy_type: str) -> str:
    """格式化回测结果为 Markdown 报告

    Args:
        raw_output: CLI 返回的 JSON 字符串
        symbol: 股票代码
        strategy_type: 策略类型
    """
    template = STRATEGY_TEMPLATES.get(strategy_type, {})
    strategy_name = template.get("name", strategy_type)

    # 解析 JSON
    try:
        data = json.loads(raw_output)
    except (json.JSONDecodeError, TypeError):
        return f"## {symbol} — {strategy_name} 回测\n\n回测执行失败，原始输出:\n{raw_output[:500]}"

    # 提取关键指标
    # quant run 返回的数据结构可能包含 strategy_report 或直接的指标
    lines = [
        f"## {symbol} — {strategy_name} 策略回测",
        "",
    ]

    # 尝试从不同数据结构中提取指标
    sharpe = _extract_value(data, ["sharpe", "sharpe_ratio"])
    max_drawdown = _extract_value(data, ["max_drawdown", "maxDrawdown"])
    win_rate = _extract_value(data, ["win_rate", "winRate"])
    profit_factor = _extract_value(data, ["profit_factor", "profitFactor"])
    total_return = _extract_value(data, ["total_return", "totalReturn", "return"])
    total_trades = _extract_value(data, ["total_trades", "totalTrades", "trades"])
    avg_profit = _extract_value(data, ["avg_profit", "avgProfit", "avg_trade"])

    has_metrics = any(v is not None for v in [sharpe, max_drawdown, win_rate, profit_factor])

    if has_metrics:
        lines.append("| 指标 | 值 |")
        lines.append("|------|-----|")
        if total_return is not None:
            lines.append(f"| 总收益 | {total_return} |")
        if sharpe is not None:
            lines.append(f"| 夏普比率 | {sharpe} |")
        if max_drawdown is not None:
            lines.append(f"| 最大回撤 | {max_drawdown} |")
        if win_rate is not None:
            lines.append(f"| 胜率 | {win_rate} |")
        if profit_factor is not None:
            lines.append(f"| 利润因子 | {profit_factor} |")
        if total_trades is not None:
            lines.append(f"| 总交易次数 | {total_trades} |")
        if avg_profit is not None:
            lines.append(f"| 平均收益 | {avg_profit} |")
    else:
        # 没有结构化指标，直接展示原始数据
        if isinstance(data, dict):
            # 可能是指标模式的输出
            lines.append("回测结果（原始数据）:")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(data, ensure_ascii=False, indent=2)[:2000])
            lines.append("```")
        elif isinstance(data, list):
            lines.append(f"返回 {len(data)} 条数据记录")
            if data:
                lines.append("")
                lines.append("```json")
                lines.append(json.dumps(data[:5], ensure_ascii=False, indent=2)[:2000])
                lines.append("```")
        else:
            lines.append(f"原始输出:\n{str(data)[:1000]}")

    return "\n".join(lines)


def _extract_value(data, keys: list) -> str | None:
    """从嵌套字典中提取值"""
    if not isinstance(data, dict):
        return None

    # 直接查找
    for key in keys:
        if key in data:
            val = data[key]
            if val is not None:
                return val

    # 在嵌套结构中查找
    for key, val in data.items():
        if isinstance(val, dict):
            for k in keys:
                if k in val and val[k] is not None:
                    return val[k]

    return None
