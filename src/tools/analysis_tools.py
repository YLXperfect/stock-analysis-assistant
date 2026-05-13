"""分析工具 — Agent 2/3 专用的高级分析 @tool

封装 src/analysis/ 下的分析引擎为 LangChain @tool。
"""

import json
from langchain_core.tools import tool

from src.tools.longbridge_sdk import (
    get_quote, get_kline, get_valuation, get_capital_flow, get_news,
    get_financial_report, get_calc_index, get_dividend, get_positions, get_portfolio,
    run_backtest as sdk_run_backtest,
)
from src.analysis.fundamental import (
    analyze_fundamentals, format_fundamental_report,
)
from src.analysis.strategy import (
    analyze_entry_strategy, analyze_dca_strategy,
    generate_bull_bear_debate, calc_stop_loss,
    format_trade_proposal, format_dca_report,
)
from src.analysis.position import (
    evaluate_position_safety, evaluate_portfolio_safety,
    format_position_report, format_portfolio_report,
)


def _call(tool_func, *args, **kwargs) -> str:  # type: ignore[no-redef]
    """调用 LangChain @tool 函数，支持位置参数自动映射"""
    import inspect
    sig = inspect.signature(tool_func.func if hasattr(tool_func, 'func') else tool_func)  # type: ignore[union-attr]
    param_names = [p for p in sig.parameters if p != 'self']
    for i, arg in enumerate(args):
        if i < len(param_names):
            kwargs[param_names[i]] = arg
    return tool_func.invoke(kwargs)  # type: ignore[union-attr]


def _safe_parse_json(raw: str, context: str = "") -> tuple:
    """安全解析 JSON，返回 (data, error_msg)"""
    if not raw or raw.startswith("SDK 错误"):
        return None, f"{context}: SDK 调用失败 - {raw}"
    try:
        return json.loads(raw), ""
    except json.JSONDecodeError as e:
        return None, f"{context}: JSON 解析失败 - {str(e)[:100]}"


@tool
def analyze_fundamental(symbol: str) -> str:
    """深度基本面分析（营收增长、利润率、ROE、负债率、现金流质量、估值水平）

    自动获取财务报表、估值、分红等数据并综合评分。

    Args:
        symbol: 股票代码
    """
    financial = _call(get_financial_report, symbol, kind="ALL")
    valuation = _call(get_valuation, symbol)
    calc_index = _call(get_calc_index, symbol, fields="pe,pb,dps_rate,turnover_rate,mktcap")
    dividend = _call(get_dividend, symbol)

    result = analyze_fundamentals(
        financial_report_json=financial,
        valuation_json=valuation,
        calc_index_json=calc_index,
        dividend_json=dividend,
    )
    return format_fundamental_report(result, symbol)


@tool
def analyze_strategy(symbol: str, strategy_type: str = "entry") -> str:
    """交易策略分析（入场时机、离场策略、含多空辩论）

    基于技术面+基本面，给出具体的买卖建议和结构化交易提案。

    Args:
        symbol: 股票代码
        strategy_type: entry(入场) / exit(离场) / dca(定投)
    """
    import pandas as pd
    from src.analysis.technical import enrich_candlesticks, analyze_technicals

    quote_json = _call(get_quote, symbol)
    kline_json = _call(get_kline, symbol, period="day", count=100)
    capital_json = _call(get_capital_flow, symbol)
    news_json = _call(get_news, symbol, count=5)

    # 解析行情
    current_price = 0
    try:
        qdata = json.loads(quote_json)
        items = qdata if isinstance(qdata, list) else qdata.get("data", [qdata])
        if items and isinstance(items[0], dict):
            current_price = float(items[0].get("last", items[0].get("last_done", 0)))
    except Exception:
        pass

    # 解析技术分析
    tech_result = {"score": 5, "trend": "数据不足", "support": 0, "resistance": 0,
                   "rsi_value": 50, "macd_signal": "", "summary": ""}
    atr = 0
    try:
        kdata = json.loads(kline_json)
        records = kdata if isinstance(kdata, list) else kdata.get("candlesticks", kdata.get("data", []))
        if records:
            df = pd.DataFrame(records)
            col_map = {}
            for col in df.columns:
                cl = col.lower()
                if cl in ("close", "close_price"): col_map[col] = "close"
                elif cl in ("open", "open_price"): col_map[col] = "open"
                elif cl in ("high", "high_price"): col_map[col] = "high"
                elif cl in ("low", "low_price"): col_map[col] = "low"
                elif cl == "volume": col_map[col] = "volume"
            df = df.rename(columns=col_map)
            if "close" in df.columns:
                df["close"] = pd.to_numeric(df["close"], errors="coerce")
                if "high" in df.columns: df["high"] = pd.to_numeric(df["high"], errors="coerce")
                if "low" in df.columns: df["low"] = pd.to_numeric(df["low"], errors="coerce")
                df = enrich_candlesticks(df)
                tech_result = analyze_technicals(df)
                last = df.iloc[-1]
                atr = float(last.get("atr14", 0))
    except Exception:
        pass

    support = tech_result.get("support", 0)
    resistance = tech_result.get("resistance", 0)
    tech_score = tech_result.get("score", 5)

    # 基本面评分（简化）
    fund_score = 5
    try:
        val_json = _call(get_valuation, symbol)
        fund_data = analyze_fundamentals(valuation_json=val_json)
        fund_score = fund_data["score"]
    except Exception:
        pass

    # 新闻摘要
    news_summary = ""
    try:
        ndata = json.loads(news_json)
        if isinstance(ndata, list):
            news_summary = "; ".join(n.get("title", "")[:50] for n in ndata[:3] if isinstance(n, dict))
    except Exception:
        pass

    # 多空辩论
    debate = generate_bull_bear_debate(
        tech_result=dict(tech_result),
        fundamental_score=fund_score,
        capital_flow_json=capital_json,
        news_summary=news_summary,
    )

    if strategy_type == "dca":
        result = analyze_dca_strategy(current_price, support, resistance)
        return format_dca_report(result, symbol)

    # 入场/离场策略
    proposal = analyze_entry_strategy(
        current_price=current_price,
        support=support,
        resistance=resistance,
        atr=atr,
        tech_score=tech_score,
        fundamental_score=fund_score,
    )
    proposal["bull_case"] = debate["bull_case"]
    proposal["bear_case"] = debate["bear_case"]

    return format_trade_proposal(proposal, debate, symbol)


@tool
def analyze_dca(symbol: str, budget: float = 10000, intervals: int = 3) -> str:
    """定投策略分析（分批建仓建议）

    Args:
        symbol: 股票代码
        budget: 总预算金额
        intervals: 分几批买入，默认3批
    """
    from src.analysis.technical import enrich_candlesticks, analyze_technicals
    import pandas as pd

    quote_json = _call(get_quote, symbol)
    kline_json = _call(get_kline, symbol, period="day", count=100)

    current_price = 0
    try:
        qdata = json.loads(quote_json)
        items = qdata if isinstance(qdata, list) else qdata.get("data", [qdata])
        if items and isinstance(items[0], dict):
            current_price = float(items[0].get("last", items[0].get("last_done", 0)))
    except Exception:
        pass

    support = 0
    try:
        kdata = json.loads(kline_json)
        records = kdata if isinstance(kdata, list) else kdata.get("candlesticks", kdata.get("data", []))
        if records:
            df = pd.DataFrame(records)
            col_map = {}
            for col in df.columns:
                cl = col.lower()
                if cl in ("close", "close_price"): col_map[col] = "close"
                elif cl in ("high", "high_price"): col_map[col] = "high"
                elif cl in ("low", "low_price"): col_map[col] = "low"
            df = df.rename(columns=col_map)
            if "close" in df.columns:
                df["close"] = pd.to_numeric(df["close"], errors="coerce")
                df = enrich_candlesticks(df)
                result = analyze_technicals(df)
                support = result.get("support", 0)
    except Exception:
        pass

    result = analyze_dca_strategy(current_price, support, current_price * 1.02, budget, intervals)
    return format_dca_report(result, symbol)


@tool
def analyze_position_safety(symbol: str = "") -> str:
    """持仓安全性分析（仓位大小评估、止损建议、风险因素识别）

    分析单个持仓或全部持仓的仓位安全性。

    Args:
        symbol: 股票代码，为空则分析全部持仓
    """
    from src.analysis.technical import enrich_candlesticks, analyze_technicals
    from src.memory.profile import load_profile
    import pandas as pd

    profile = load_profile()

    # 获取持仓
    positions_json = _call(get_positions)
    portfolio_json = _call(get_portfolio)

    try:
        positions_data = json.loads(positions_json)
    except Exception:
        return "无法获取持仓数据"

    if isinstance(positions_data, dict):
        positions_data = positions_data.get("data", positions_data.get("items", []))
    if not isinstance(positions_data, list):
        positions_data = []

    # 组合总资产
    portfolio_total = 0
    try:
        pdata = json.loads(portfolio_json)
        if isinstance(pdata, dict):
            portfolio_total = float(pdata.get("total_assets", pdata.get("net_assets", 0)))
        elif isinstance(pdata, list) and pdata:
            portfolio_total = float(pdata[0].get("total_assets", 0))
    except Exception:
        pass

    if not positions_data:
        return "当前无持仓"

    # 筛选
    if symbol:
        positions_data = [p for p in positions_data if symbol.upper() in p.get("symbol", "").upper()]
        if not positions_data:
            return f"未找到 {symbol} 的持仓"

    # 分析每个持仓
    analyzed = []
    for pos in positions_data[:20]:  # 限制数量
        sym = pos.get("symbol", "?")
        qty = float(pos.get("quantity", pos.get("qty", 0)))
        cost = float(pos.get("cost_price", pos.get("avg_cost", 0)))
        current = float(pos.get("market_price", pos.get("current_price", 0)))
        if current == 0:
            # 需要获取实时价格
            q = _call(get_quote, sym)
            try:
                qd = json.loads(q)
                items = qd if isinstance(qd, list) else qd.get("data", [qd])
                if items and isinstance(items[0], dict):
                    current = float(items[0].get("last", items[0].get("last_done", 0)))
            except Exception:
                pass

        # 技术面评分
        tech_score = 5
        tech_trend = ""
        atr = 0
        try:
            kline_json = _call(get_kline, sym, period="day", count=60)
            kdata = json.loads(kline_json)
            records = kdata if isinstance(kdata, list) else kdata.get("candlesticks", kdata.get("data", []))
            if records:
                df = pd.DataFrame(records)
                col_map = {}
                for col in df.columns:
                    cl = col.lower()
                    if cl in ("close", "close_price"): col_map[col] = "close"
                    elif cl in ("high", "high_price"): col_map[col] = "high"
                    elif cl in ("low", "low_price"): col_map[col] = "low"
                df = df.rename(columns=col_map)
                if "close" in df.columns:
                    df["close"] = pd.to_numeric(df["close"], errors="coerce")
                    if "high" in df.columns: df["high"] = pd.to_numeric(df["high"], errors="coerce")
                    if "low" in df.columns: df["low"] = pd.to_numeric(df["low"], errors="coerce")
                    df = enrich_candlesticks(df)
                    t = analyze_technicals(df)
                    tech_score = t["score"]
                    tech_trend = t["trend"]
                    atr = float(df.iloc[-1].get("atr14", 0))
        except Exception:
            pass

        if portfolio_total == 0:
            portfolio_total = sum(float(p.get("market_value", p.get("position_value", qty * current)))
                                  for p in positions_data)

        analyzed.append({
            "symbol": sym, "quantity": qty, "cost_price": cost,
            "current_price": current, "tech_score": tech_score,
            "tech_trend": tech_trend, "atr": atr,
        })

    results = evaluate_portfolio_safety(
        analyzed, portfolio_total,
        risk_tolerance=profile.risk_tolerance,
        max_position_pct=profile.max_position_pct,
    )

    if symbol and len(results) == 1:
        return format_position_report(results[0])
    return format_portfolio_report(results)


@tool
def analyze_portfolio() -> str:
    """投资组合全面分析（持仓概览、安全评估、配置建议）"""
    return analyze_position_safety.invoke({"symbol": ""})


# ── 选股工具 ────────────────────────────────────────────────────


@tool
def get_stock_universe(market: str = "US") -> str:
    """获取常用股票池列表，用于选股筛选

    返回按市场分类的常用股票代码列表（美股约50只、港股约30只）。

    Args:
        market: "US" 或 "HK"
    """
    from pathlib import Path

    universe_file = Path(__file__).parent.parent.parent / "data" / "stock_universe.json"
    if not universe_file.exists():
        return "股票池文件不存在: data/stock_universe.json"

    try:
        with open(universe_file, encoding="utf-8") as f:
            data = json.loads(f.read())
    except Exception as e:
        return f"读取股票池失败: {e}"

    market = market.upper()
    if market == "US":
        stocks = data.get("US", [])
    elif market == "HK":
        stocks = data.get("HK", [])
    else:
        stocks = data.get("US", []) + data.get("HK", [])

    if not stocks:
        return f"未找到 {market} 市场的股票"

    lines = [f"## {market} 股票池（{len(stocks)} 只）", ""]
    for s in stocks:
        lines.append(f"- {s['symbol']} ({s['name']}) — {s['sector']}")
    return "\n".join(lines)


@tool
def quick_screen_stock(symbol: str) -> str:
    """快速获取一只股票的核心筛选指标（PE/PB/市值/ROE/营收增长/股息率）

    一次调用获取多维度数据，用于批量选股筛选。

    Args:
        symbol: 股票代码
    """
    idx_raw = _call(get_calc_index, symbol, fields="pe,pb,dps_rate,turnover_rate,mktcap,roe")
    val_raw = _call(get_valuation, symbol)

    result = {"symbol": symbol, "pe": "-", "pb": "-", "mktcap": "-",
              "roe": "-", "dps_rate": "-", "revenue_growth": "-"}

    # 解析 calc-index
    try:
        idata = json.loads(idx_raw)
        items = idata if isinstance(idata, list) else idata.get("data", [idata])
        if items and isinstance(items[0], dict):
            d = items[0]
            for key in ["pe", "pb", "mktcap", "roe", "dps_rate", "turnover_rate"]:
                if key in d and d[key] is not None:
                    result[key] = d[key]
    except Exception:
        pass

    # 解析 valuation（补充 revenue_growth）
    try:
        vdata = json.loads(val_raw)
        items = vdata if isinstance(vdata, list) else vdata.get("data", [vdata])
        if items and isinstance(items[0], dict):
            d = items[0]
            # 尝试获取各种增长字段
            for field in ["revenue_growth", "revenue_yoy_growth", "income_growth"]:
                if field in d and d[field] is not None:
                    result["revenue_growth"] = d[field]
                    break
    except Exception:
        pass

    lines = [
        f"## {symbol} 快速指标",
        "",
        f"| 指标 | 值 |",
        f"|------|-----|",
        f"| PE | {result['pe']} |",
        f"| PB | {result['pb']} |",
        f"| 市值 | {result['mktcap']} |",
        f"| ROE | {result['roe']} |",
        f"| 股息率 | {result['dps_rate']} |",
        f"| 营收增长 | {result['revenue_growth']} |",
    ]
    return "\n".join(lines)


# ── 回测工具 ────────────────────────────────────────────────────


@tool
def analyze_backtest(symbol: str, start: str, end: str, strategy_type: str = "ma_cross",
                     fast_period: int = 5, slow_period: int = 20,
                     rsi_period: int = 14, oversold: float = 30.0, overbought: float = 70.0,
                     boll_period: int = 20, boll_stddev: float = 2.0) -> str:
    """运行策略回测并生成分析报告

    内置 4 种常用策略模板，自动生成 PineScript 并执行回测。

    Args:
        symbol: 股票代码，如 TSLA.US, 700.HK
        start: 回测开始日期 YYYY-MM-DD
        end: 回测结束日期 YYYY-MM-DD
        strategy_type: 策略类型 ma_cross/rsi/bollinger/macd
        fast_period: 均线快线周期（ma_cross 策略），默认 5
        slow_period: 均线慢线周期（ma_cross 策略），默认 20
        rsi_period: RSI 周期（rsi 策略），默认 14
        oversold: RSI 超卖线（rsi 策略），默认 30
        overbought: RSI 超买线（rsi 策略），默认 70
        boll_period: 布林带周期（bollinger 策略），默认 20
        boll_stddev: 布林带标准差（bollinger 策略），默认 2.0
    """
    from src.analysis.backtest import generate_pine_script, format_backtest_result

    # 生成 PineScript
    params = {
        "fast_period": fast_period, "slow_period": slow_period,
        "rsi_period": rsi_period, "oversold": oversold, "overbought": overbought,
        "boll_period": boll_period, "boll_stddev": boll_stddev,
    }

    try:
        script = generate_pine_script(strategy_type, **params)
    except ValueError as e:
        return str(e)

    # 执行回测（使用MCP）
    raw = _call(sdk_run_backtest, symbol=symbol, start=start, end=end,
                        script=script, period="day")

    return format_backtest_result(raw, symbol, strategy_type)
