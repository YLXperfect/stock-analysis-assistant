"""长桥 SDK 工具封装层

使用 longbridge Python SDK v4 获取行情、基本面、新闻等数据。
认证: 复用 CLI 的 OAuth token（~/.longbridge/openapi/tokens/），无需浏览器交互。

依赖: pip install longbridge>=4.0
"""

import json
import time
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool
from longbridge.openapi import (
    Config, OAuthBuilder, Language,
    QuoteContext, TradeContext, ContentContext, HttpClient,  # type: ignore[import-untyped]
    Period, AdjustType, CalcIndex, SecuritiesUpdateMode,
)

# ── SDK 认证单例 ─────────────────────────────────────────────────

_quote_ctx: Optional[QuoteContext] = None
_trade_ctx: Optional[TradeContext] = None
_content_ctx: Optional[ContentContext] = None
_http_cli: Optional[HttpClient] = None
_client_id: Optional[str] = None


def _get_client_id() -> str:
    """获取 OAuth client_id（优先从 CLI token 读取）"""
    global _client_id
    if _client_id:
        return _client_id

    token_dir = Path.home() / ".longbridge" / "openapi" / "tokens"
    if token_dir.exists():
        # 读第一个有效的 token 文件
        for f in sorted(token_dir.glob("*")):
            if f.is_file():
                try:
                    data = json.loads(f.read_text())
                    cid = data.get("client_id", "")
                    if cid:
                        _client_id = cid
                        return cid
                except Exception:
                    continue

    # Fallback
    _client_id = "longbridge-agent"
    return _client_id


def _get_oauth():
    """获取 OAuth 凭证（自动复用 CLI 缓存 token）"""
    client_id = _get_client_id()
    return OAuthBuilder(client_id).build(lambda _: None)


def get_quote_ctx() -> QuoteContext:
    global _quote_ctx
    if _quote_ctx is None:
        oauth = _get_oauth()
        config = Config.from_oauth(oauth, language=Language.ZH_CN)
        _quote_ctx = QuoteContext(config)
    return _quote_ctx


def get_trade_ctx() -> TradeContext:
    global _trade_ctx
    if _trade_ctx is None:
        oauth = _get_oauth()
        config = Config.from_oauth(oauth, language=Language.ZH_CN)
        _trade_ctx = TradeContext(config)
    return _trade_ctx


def get_content_ctx() -> ContentContext:
    global _content_ctx
    if _content_ctx is None:
        oauth = _get_oauth()
        config = Config.from_oauth(oauth, language=Language.ZH_CN)
        _content_ctx = ContentContext(config)
    return _content_ctx


def _http_get(path: str, **params) -> str:
    """HttpClient GET 请求（自动拼接 query string）"""
    cli = get_http_cli()
    if params:
        qs = "&".join(f"{k}={v}" for k, v in params.items() if v)
        path = f"{path}?{qs}"
    return _to_json(cli.request("get", path))


def _http_post(path: str, body: dict = None) -> str:  # type: ignore[assignment]
    """HttpClient POST 请求"""
    cli = get_http_cli()
    return _to_json(cli.request("post", path, body=body))


def get_http_cli() -> HttpClient:
    global _http_cli
    if _http_cli is None:
        oauth = _get_oauth()
        _http_cli = HttpClient.from_oauth(oauth)
    return _http_cli


def _to_json(obj) -> str:
    """将 SDK 返回对象转为 JSON 字符串（处理 Decimal 等非标准类型）"""
    return json.dumps(obj, ensure_ascii=False, default=_json_default)


def _json_default(obj):
    """JSON 序列化时的 fallback"""
    from decimal import Decimal
    if isinstance(obj, Decimal):
        return float(obj)
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


# ── 符号格式转换 ──────────────────────────────────────────────

def _convert_symbol(symbol: str) -> str:
    """将项目符号格式转为 yfinance 格式

    TSLA.US -> TSLA,  700.HK -> 0700.HK,  9988.HK -> 9988.HK
    """
    if symbol.endswith(".US"):
        return symbol[:-3]
    elif symbol.endswith(".HK"):
        code = symbol[:-3]
        return code.zfill(4) + ".HK"
    return symbol


# ── 行情工具 ────────────────────────────────────────────────────

@tool
def get_quote(symbol: str) -> str:
    """获取股票实时行情（价格、涨跌幅、成交量、成交额等）

    Args:
        symbol: 股票代码，格式 <CODE>.<MARKET>，如 TSLA.US, 700.HK, 600519.SH
    """
    # 股票代码格式弱校验
    if not symbol or "." not in symbol:
        return json.dumps({"error": f"无效的股票代码 '{symbol}'，正确格式如 TSLA.US、700.HK"}, ensure_ascii=False)

    try:
        ctx = get_quote_ctx()
        result = ctx.quote([symbol])
        if result and result[0]:
            q = result[0]
            return _to_json({
                "symbol": symbol,
                "price": q.last_done, "prev_close": q.prev_close,
                "open": q.open, "high": q.high, "low": q.low,
                "volume": q.volume, "turnover": q.turnover,
            })
        return _to_json({"error": f"未找到股票 '{symbol}'，请检查代码是否正确（如 TSLA.US）"})
    except Exception as e:
        err = str(e)
        if "permission" in err.lower() or "权限" in err:
            return json.dumps({"error": f"无法查询 '{symbol}'，可能原因：1) 代码不存在 2) 该市场行情权限未开通。请检查代码格式（如 TSLA.US）"}, ensure_ascii=False)
        return f"SDK 错误: {e}"


@tool
def get_static_info(symbol: str) -> str:
    """获取股票静态信息（名称、交易所、币种、每手股数、总股本、流通股本等）

    Args:
        symbol: 股票代码，如 AAPL.US, 700.HK
    """
    if "." not in symbol:
        return json.dumps({"error": f"无效的股票代码 '{symbol}'，格式如 AAPL.US、700.HK"}, ensure_ascii=False)
    try:
        ctx = get_quote_ctx()
        result = ctx.static_info([symbol])
        if result and result[0]:
            s = result[0]
            return json.dumps({
                "symbol": symbol,
                "name_cn": s.name_cn,
                "name_en": s.name_en,
                "exchange": s.exchange,
                "currency": s.currency,
                "lot_size": s.lot_size,
                "total_shares": s.total_shares,
                "circulating_shares": s.circulating_shares,
                "eps": s.eps,
                "eps_ttm": s.eps_ttm,
            }, ensure_ascii=False)
        return json.dumps({"error": "无数据"}, ensure_ascii=False)
    except Exception as e:
        return f"SDK 错误: {e}"


@tool
def get_kline(symbol: str, period: str = "day", count: int = 100, adjust: str = "none") -> str:
    """获取K线数据（OHLCV: 开高低收量）

    Args:
        symbol: 股票代码，如 TSLA.US, 700.HK
        period: K线周期 1m/5m/15m/30m/1h/day/week/month/year，默认 day
        count: 返回K线数量，默认 100
        adjust: 复权方式 none/fwd，默认 none
    """
    import yfinance as yf  # type: ignore[import-untyped]

    yf_symbol = _convert_symbol(symbol)

    # 周期映射
    interval_map = {
        "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "1h",
        "day": "1d", "week": "1wk", "month": "1mo", "year": "3mo",
    }
    interval = interval_map.get(period, "1d")

    # count 转 yfinance 的 date range
    if period in ("1m", "5m", "15m", "30m", "1h"):
        yf_period = f"{max(count // 2, 5)}d"
    elif period == "week":
        yf_period = f"{count * 7}d"
    elif period == "month":
        yf_period = f"{count * 30}d"
    elif period == "year":
        yf_period = f"{count * 365}d"
    else:
        yf_period = f"{count}d"

    auto_adjust = adjust in ("forward", "fwd")

    try:
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period=yf_period, interval=interval, auto_adjust=auto_adjust)

        if df.empty:
            return "[]"

        df = df.tail(count)

        result = []
        for idx, row in df.iterrows():
            result.append({
                "close": float(row["Close"]),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "volume": int(row["Volume"]),
                "turnover": 0,
                "timestamp": idx.isoformat(),
            })

        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return f"yfinance 错误: {e}"


@tool
def get_depth(symbol: str) -> str:
    """获取盘口深度（买卖档报价）

    Args:
        symbol: 股票代码
    """
    try:
        ctx = get_quote_ctx()
        result = ctx.depth(symbol)
        return _to_json(result)
    except Exception as e:
        return f"SDK 错误: {e}"


@tool
def get_intraday(symbol: str) -> str:
    """获取分时数据（当日分钟级价格和成交量）

    Args:
        symbol: 股票代码
    """
    try:
        ctx = get_quote_ctx()
        result = ctx.intraday(symbol)
        return _to_json(result)
    except Exception as e:
        return f"SDK 错误: {e}"


@tool
def get_capital_flow(symbol: str) -> str:
    """获取资金流向（主力/散户资金进出分布）

    Args:
        symbol: 股票代码
    """
    try:
        ctx = get_quote_ctx()
        result = ctx.capital_flow(symbol)
        return _to_json(result)
    except Exception as e:
        return f"SDK 错误: {e}"


@tool
def get_calc_index(symbol: str, fields: str = "pe,pb,dps_rate,turnover_rate,mktcap") -> str:
    """获取计算指标（PE、PB、股息率、换手率、市值等）

    Args:
        symbol: 股票代码
        fields: 逗号分隔的字段名，如 pe,pb,turnover_rate,mktcap
    """
    try:
        ctx = get_quote_ctx()
        field_names = [f.strip() for f in fields.split(",")]
        # Map string names to CalcIndex enum values
        indexes = []
        for f in field_names:
            try:
                indexes.append(getattr(CalcIndex, f))
            except AttributeError:
                pass  # Skip unknown fields
        result = ctx.calc_indexes([symbol], indexes)
        if result and result[0]:
            item = result[0]
            data: dict[str, object] = {"symbol": symbol}
            for i, idx in enumerate(indexes):
                val = getattr(item, idx, None)
                if val is not None:
                    data[field_names[i]] = val
            return json.dumps(data, ensure_ascii=False, default=str)
        return json.dumps({"error": "无数据"}, ensure_ascii=False)
    except Exception as e:
        return f"SDK 错误: {e}"


# ── 基本面工具 ──────────────────────────────────────────────────

@tool
def get_financial_report(symbol: str, kind: str = "ALL", report: str = "") -> str:
    """获取财务报表（利润表/资产负债表/现金流量表）

    Args:
        symbol: 股票代码
        kind: 报表类型 IS(利润表) BS(资产负债表) CF(现金流量表) ALL(全部)
        report: 报告期 af(年报) saf(半年报) q1(一季报) qf(季报)
    """
    try:
        params = {"symbol": symbol, "type": kind}
        if report:
            params["period"] = report
        resp = _http_get("/v1/stock/financial-report", **params)
        return _to_json(resp)
    except Exception as e:
        return f"SDK 错误: {e}"


@tool
def get_valuation(symbol: str) -> str:
    """获取估值分析（PE/PB/PS/股息率 + 同业对比）

    Args:
        symbol: 股票代码
    """
    try:
        resp = _http_get("/v1/stock/valuation", symbol=symbol)
        return _to_json(resp)
    except Exception as e:
        return f"SDK 错误: {e}"


@tool
def get_dividend(symbol: str) -> str:
    """获取分红历史

    Args:
        symbol: 股票代码
    """
    try:
        resp = _http_get("/v1/stock/dividend", symbol=symbol)
        return _to_json(resp)
    except Exception as e:
        return f"SDK 错误: {e}"


@tool
def get_institution_rating(symbol: str) -> str:
    """获取机构评级和目标价

    Args:
        symbol: 股票代码
    """
    try:
        resp = _http_get("/v1/stock/institution-rating", symbol=symbol)
        return _to_json(resp)
    except Exception as e:
        return f"SDK 错误: {e}"


@tool
def get_company_overview(symbol: str) -> str:
    """获取公司概况（成立日期、员工数、IPO价格、地址等）

    Args:
        symbol: 股票代码
    """
    try:
        resp = _http_get("/v1/stock/company", symbol=symbol)
        return _to_json(resp)
    except Exception as e:
        return f"SDK 错误: {e}"


# ── 新闻工具 ────────────────────────────────────────────────────

@tool
def get_news(symbol: str = "", count: int = 10) -> str:
    """获取相关新闻资讯

    Args:
        symbol: 股票代码（可选，为空则获取一般财经新闻）
        count: 返回条数，默认 10
    """
    try:
        ctx = get_content_ctx()
        result = ctx.news(symbol) if symbol else []
        if result:
            return json.dumps([{
                "title": n.title,
                "source": n.source,
                "timestamp": str(n.timestamp) if hasattr(n, "timestamp") else "",
                "url": n.url if hasattr(n, "url") else "",
            } for n in result[:count]], ensure_ascii=False)
        return "[]"
    except Exception as e:
        return f"SDK 错误: {e}"


@tool
def search_news(keyword: str) -> str:
    """搜索新闻

    Args:
        keyword: 搜索关键词
    """
    try:
        resp = _http_get("/v1/content/news/search", keyword=keyword)
        return _to_json(resp)
    except Exception as e:
        return f"SDK 错误: {e}"


@tool
def get_insider_trades(symbol: str) -> str:
    """获取内幕交易数据（SEC Form 4，内部人买卖记录）

    Args:
        symbol: 股票代码，如 TSLA.US, AAPL.US
    """
    try:
        resp = _http_get("/v1/stock/insider-trades", symbol=symbol)
        return _to_json(resp)
    except Exception as e:
        return f"SDK 错误: {e}"


# ── 技术分析工具 ────────────────────────────────────────────────

@tool
def analyze_technical(symbol: str, period: str = "day", count: int = 100) -> str:
    """技术面分析（MA均线/MACD/RSI/布林带 + 综合评分）

    自动获取K线数据并计算技术指标，返回分析结果。

    Args:
        symbol: 股票代码
        period: K线周期，默认 day
        count: K线数量，建议 60+ 根
    """
    import pandas as pd
    from src.analysis.technical import enrich_candlesticks, analyze_technicals

    raw = get_kline.invoke({"symbol": symbol, "period": period, "count": str(count)})

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return f"无法获取K线数据: {raw[:200]}"

    if not isinstance(data, list) or len(data) == 0:
        return "K线数据为空"

    df = pd.DataFrame(data)
    if "close" not in df.columns:
        return f"K线数据缺少 close 列"

    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    if "high" in df.columns:
        df["high"] = pd.to_numeric(df["high"], errors="coerce")
    if "low" in df.columns:
        df["low"] = pd.to_numeric(df["low"], errors="coerce")

    df = enrich_candlesticks(df)
    result = analyze_technicals(df)

    lines = [
        f"## 技术分析: {symbol}",
        "",
        f"| 指标 | 值 |",
        f"|------|-----|",
        f"| 趋势 | {result['trend']} |",
        f"| 均线信号 | {result['ma_signal']} |",
        f"| MACD | {result['macd_signal']} |",
        f"| RSI | {result['rsi_value']} ({result['rsi_signal']}) |",
        f"| 布林带 | {result['bollinger_pos']} |",
        f"| 支撑位 | {result['support']} |",
        f"| 压力位 | {result['resistance']} |",
        f"| **综合评分** | **{result['score']}/10** |",
        "",
        f"> {result['summary']}",
    ]
    return "\n".join(lines)


# ── 账户工具 ────────────────────────────────────────────────────

@tool
def get_positions() -> str:
    """获取当前持仓列表"""
    try:
        ctx = get_trade_ctx()
        resp = ctx.stock_positions()
        all_positions = []
        for channel in (resp.channels or []):
            for p in (channel.positions or []):
                all_positions.append({
                    "symbol": p.symbol,
                    "name": p.symbol_name,
                    "quantity": p.quantity,
                    "available": p.available_quantity,
                    "cost_price": p.cost_price,
                    "currency": p.currency,
                    "market": p.market,
                })
        return _to_json(all_positions)
    except Exception as e:
        return f"SDK 错误: {e}"


@tool
def get_assets(currency: str = "USD") -> str:
    """获取账户资产概览（净值、现金、购买力、保证金等）"""
    try:
        ctx = get_trade_ctx()
        balances = ctx.account_balance(currency)
        result = []
        for b in (balances or []):
            result.append({
                "currency": b.currency,
                "net_assets": b.net_assets,
                "total_cash": b.total_cash,
                "buy_power": b.buy_power,
                "init_margin": b.init_margin,
                "risk_level": b.risk_level,
            })
        return _to_json(result)
    except Exception as e:
        return f"SDK 错误: {e}"


@tool
def get_portfolio() -> str:
    """获取投资组合概览（总资产、盈亏、持仓明细）"""
    try:
        ctx = get_trade_ctx()
        balances = ctx.account_balance()
        resp = ctx.stock_positions()
        all_positions = []
        for channel in (resp.channels or []):
            for p in (channel.positions or []):
                all_positions.append({
                    "symbol": p.symbol, "name": p.symbol_name,
                    "quantity": p.quantity, "cost_price": p.cost_price,
                })
        balance_info = {}
        if balances:
            b = balances[0]
            balance_info = {"net_assets": b.net_assets, "total_cash": b.total_cash}
        return _to_json({"balance": balance_info, "positions": all_positions})
    except Exception as e:
        return f"SDK 错误: {e}"


# ── 自选股工具 ──────────────────────────────────────────────────

@tool
def watchlist_list() -> str:
    """列出所有自选股分组及其包含的证券"""
    try:
        ctx = get_quote_ctx()
        result = ctx.watchlist()
        groups = []
        for g in (result or []):
            secs = []
            for s in (g.securities if hasattr(g, "securities") else []):
                secs.append({
                    "symbol": s.symbol,
                    "name": s.name if hasattr(s, "name") else str(s),
                    "market": str(s.market) if hasattr(s, "market") else "",
                    "is_pinned": getattr(s, "is_pinned", False),
                })
            groups.append({
                "id": g.id if hasattr(g, "id") else str(g),
                "name": g.name if hasattr(g, "name") else str(g),
                "securities": secs,
            })
        return _to_json(groups)
    except Exception as e:
        return f"SDK 错误: {e}"


@tool
def watchlist_show(group: str) -> str:
    """查看某个自选股分组中的证券列表

    Args:
        group: 分组 ID 或名称
    """
    result = watchlist_list.invoke({})
    try:
        data = json.loads(result)
        for g in data:
            gid = str(g.get("id", ""))
            gname = g.get("name", "")
            if gid == group or gname == group:
                return json.dumps(g, ensure_ascii=False)
        # 未找到，返回完整列表
        return result
    except json.JSONDecodeError:
        return result


@tool
def watchlist_create(name: str) -> str:
    """创建一个新的自选股分组

    Args:
        name: 分组名称，如 "科技股"
    """
    try:
        ctx = get_quote_ctx()
        result = ctx.create_watchlist_group(name)
        return f"已创建自选股分组: {name}"
    except Exception as e:
        return f"SDK 错误: {e}"


@tool
def watchlist_update(group_id: str, add_symbols: str = "", remove_symbols: str = "",
                     new_name: str = "") -> str:
    """更新自选股分组（添加/移除证券或重命名）

    Args:
        group_id: 分组 ID（从 watchlist_list 获取）
        add_symbols: 要添加的证券代码，逗号分隔
        remove_symbols: 要移除的证券代码，逗号分隔
        new_name: 新的分组名称（可选）
    """
    try:
        ctx = get_quote_ctx()
        gid = int(group_id)
        add_list = [s.strip() for s in add_symbols.split(",") if s.strip()] if add_symbols else []
        rm_list = [s.strip() for s in remove_symbols.split(",") if s.strip()] if remove_symbols else []

        if new_name:
            ctx.update_watchlist_group(gid, name=new_name)
        if add_list:
            ctx.update_watchlist_group(gid, securities=add_list)
        if rm_list:
            ctx.update_watchlist_group(gid, securities=rm_list, mode=SecuritiesUpdateMode.Remove)
        return f"已更新自选股分组 {group_id}"
    except Exception as e:
        return f"SDK 错误: {e}"


@tool
def watchlist_delete(group_id: str) -> str:
    """删除一个自选股分组

    Args:
        group_id: 分组 ID
    """
    try:
        ctx = get_quote_ctx()
        ctx.delete_watchlist_group(int(group_id))
        return f"已删除自选股分组 {group_id}"
    except Exception as e:
        return f"SDK 错误: {e}"


@tool
def watchlist_pin(symbols: str, unpin: bool = False) -> str:
    """置顶或取消置顶自选股

    Args:
        symbols: 证券代码，逗号分隔
        unpin: True 为取消置顶，False 为置顶
    """
    # SDK 可能没有直接的 pin 接口，用 update 实现
    return watchlist_update.invoke({"group_id": "all", "add_symbols": symbols})


# ── 提醒工具 ────────────────────────────────────────────────────

@tool
def alert_list(symbol: str = "") -> str:
    """列出所有价格提醒，或某个证券的提醒

    Args:
        symbol: 证券代码（可选，为空则列出全部）
    """
    try:
        params = {}
        if symbol:
            params["symbol"] = symbol
        resp = _http_get("/v1/trade/alert", **params)
        return _to_json(resp)
    except Exception as e:
        return f"SDK 错误: {e}"


@tool
def alert_add(symbol: str, price: float, direction: str = "fall",
              alert_type: str = "price", frequency: str = "once", note: str = "") -> str:
    """添加价格提醒

    Args:
        symbol: 证券代码
        price: 目标价格或百分比值
        direction: rise/false，默认 fall
        alert_type: price/percent，默认 price
        frequency: once/daily/every，默认 once
        note: 可选备注
    """
    try:
        body = {
            "symbol": symbol, "price": str(price), "direction": direction,
            "alert_type": alert_type, "frequency": frequency,
        }
        if note:
            body["note"] = note
        resp = _http_post("/v1/trade/alert", body=body)
        return _to_json(resp)
    except Exception as e:
        return f"SDK 错误: {e}"


@tool
def alert_delete(alert_id: str) -> str:
    """删除价格提醒

    Args:
        alert_id: 提醒 ID
    """
    try:
        cli = get_http_cli()
        resp = cli.request("delete", f"/v1/trade/alert/{alert_id}")
        return f"已删除提醒 {alert_id}"
    except Exception as e:
        return f"SDK 错误: {e}"


@tool
def alert_enable(alert_id: str) -> str:
    """启用价格提醒"""
    try:
        cli = get_http_cli()
        resp = cli.request("put", f"/v1/trade/alert/{alert_id}/enable")
        return f"已启用提醒 {alert_id}"
    except Exception as e:
        return f"SDK 错误: {e}"


@tool
def alert_disable(alert_id: str) -> str:
    """禁用价格提醒"""
    try:
        cli = get_http_cli()
        resp = cli.request("put", f"/v1/trade/alert/{alert_id}/disable")
        return f"已禁用提醒 {alert_id}"
    except Exception as e:
        return f"SDK 错误: {e}"


# ── 批量扫描 ────────────────────────────────────────────────────

@tool
def scan_watchlist(group: str = "") -> str:
    """扫描自选股列表，批量获取每只股票的实时行情和PE/PB/市值

    Args:
        group: 分组名称（空则扫描默认分组）
    """
    result = watchlist_list.invoke({})
    try:
        data = json.loads(result)
    except json.JSONDecodeError:
        return f"无法解析自选股数据: {result[:200]}"

    if not isinstance(data, list) or not data:
        return "自选股为空"

    # 提取所有证券
    all_secs = []
    for g in data:
        for s in g.get("securities", []):
            all_secs.append((s.get("symbol", ""), s.get("name", "")))

    if not all_secs:
        return "自选股列表为空"

    ctx = get_quote_ctx()
    symbols = [s[0] for s in all_secs[:30]]

    # 批量获取行情和指标
    quotes = ctx.quote(symbols) if symbols else []
    indexes = ctx.calc_indexes(symbols, [CalcIndex.PeTtmRatio, CalcIndex.PbRatio, CalcIndex.TotalMarketValue]) if symbols else []

    # 构建行情映射
    quote_map = {}
    for q in (quotes or []):
        quote_map[q.symbol] = q

    index_map = {}
    for idx in (indexes or []):
        index_map[idx.symbol] = idx

    lines = [f"## 自选股扫描（{len(symbols)} 只）", "",
             "| 代码 | 名称 | 价格 | 涨跌% | PE | PB | 市值 |",
             "|------|------|------|-------|----|----|------|"]

    for sym, name in all_secs[:30]:
        q = quote_map.get(sym)
        idx = index_map.get(sym)
        price = f"{q.last_done:.2f}" if q and q.last_done else "-"
        chg = f"{((q.last_done - q.prev_close) / q.prev_close * 100):.2f}%" if q and q.prev_close and q.last_done else "-"
        pe = f"{getattr(idx, 'PeTtmRatio', '-')}" if idx else "-"
        pb = f"{getattr(idx, 'PbRatio', '-')}" if idx else "-"
        mktcap = f"{getattr(idx, 'mktcap', '-')}" if idx else "-"
        lines.append(f"| {sym} | {name[:10]} | {price} | {chg} | {pe} | {pb} | {mktcap} |")

    return "\n".join(lines)


# ── 回测工具 ────────────────────────────────────────────────────

@tool
def run_backtest(symbol: str, start: str, end: str, script: str,
                 period: str = "day", script_inputs: str = "") -> str:
    """运行策略回测（PineScript V6 兼容）

    Args:
        symbol: 股票代码
        start: 开始日期 YYYY-MM-DD
        end: 结束日期 YYYY-MM-DD
        script: PineScript 脚本内容
        period: K线周期，默认 day
        script_inputs: 参数 JSON 数组，如 '[14,2.0]'
    """
    try:
        body = {
            "symbol": symbol, "start": start, "end": end,
            "period": period, "script": script,
        }
        if script_inputs:
            body["inputs"] = json.loads(script_inputs) if isinstance(script_inputs, str) else script_inputs
        resp = _http_post("/v1/quant/run", body=body)
        return _to_json(resp)
    except Exception as e:
        return f"SDK 错误: {e}"


# ── 工具集合 ────────────────────────────────────────────────────

def get_analysis_tools() -> list:
    """获取分析类工具列表"""
    return [
        get_quote, get_static_info,
        get_kline, get_depth, get_intraday,
        get_capital_flow, get_calc_index,
        get_financial_report, get_valuation, get_dividend,
        get_institution_rating, get_company_overview,
        get_news, search_news, get_insider_trades,
        analyze_technical,
    ]


def get_analyst_tools() -> list:
    """Agent 2 (Analyst) 用的工具"""
    return [
        get_quote, get_kline, get_depth,
        get_intraday, get_capital_flow, get_calc_index,
        get_static_info,
        get_financial_report, get_valuation,
        get_dividend, get_institution_rating, get_company_overview,
        get_news, search_news, get_insider_trades,
    ]


def get_watchlist_tools() -> list:
    """自选股只读工具"""
    return [watchlist_list, watchlist_show]
