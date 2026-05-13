"""深度基本面分析模块

基于财务报表、估值、共识等数据，输出结构化基本面评分。
不依赖 LLM，纯函数式分析。
"""

import json
from typing import TypedDict, Optional


class FundamentalResult(TypedDict):
    revenue_growth: str       # 营收增长评价
    profit_margin: str        # 净利率评价
    roe: str                  # ROE 评价
    debt_ratio: str           # 负债率评价
    cashflow_quality: str     # 现金流质量
    valuation_level: str      # 估值水平
    dividend_info: str        # 分红信息
    score: int                # 综合评分 1-10
    rating: str               # excellent / good / fair / poor
    summary: str              # 一句话总结


def _safe_float(val, default=0.0) -> float:
    """安全转换为浮点数"""
    if val is None:
        return default
    try:
        return float(str(val).replace(",", "").replace("%", ""))
    except (ValueError, TypeError):
        return default


def _score_revenue_growth(reports_json: str) -> tuple[str, int]:
    """营收增长评分 (权重15%)"""
    try:
        data = json.loads(reports_json)
        items = data if isinstance(data, list) else data.get("items", data.get("data", []))
        if not items or len(items) < 2:
            return "数据不足", 5

        # 取最近两年营收
        revenues = []
        for item in items[:4]:
            rev = None
            if isinstance(item, dict):
                for key in ("total_revenue", "revenue", "营业收入", "totalRevenue"):
                    if key in item:
                        rev = _safe_float(item[key])
                        break
            if rev and rev > 0:
                revenues.append(rev)

        if len(revenues) >= 2:
            growth = (revenues[0] - revenues[1]) / revenues[1] * 100
            if growth > 20:
                return f"营收同比增长 {growth:.1f}%，高增长", 10
            elif growth > 10:
                return f"营收同比增长 {growth:.1f}%，稳健增长", 7
            elif growth > 5:
                return f"营收同比增长 {growth:.1f}%，温和增长", 5
            elif growth > 0:
                return f"营收同比增长 {growth:.1f}%，增长放缓", 3
            else:
                return f"营收同比 {growth:.1f}%，负增长", 1
        return "营收数据有限", 5
    except Exception:
        return "无法计算", 5


def _score_profit_margin(calc_index_json: str, financial_json: str) -> tuple[str, int]:
    """净利率评分 (权重15%)"""
    try:
        # 优先从 calc_index 获取
        data = json.loads(calc_index_json)
        items = data if isinstance(data, list) else data.get("items", data.get("data", [data]))
        for item in items if isinstance(items, list) else [items]:
            if isinstance(item, dict):
                for key in ("net_profit_margin", "profit_margin", "netMargin"):
                    if key in item:
                        margin = _safe_float(item[key])
                        if margin > 0:
                            if margin > 25:
                                return f"净利率 {margin:.1f}%，优秀", 10
                            elif margin > 15:
                                return f"净利率 {margin:.1f}%，良好", 7
                            elif margin > 5:
                                return f"净利率 {margin:.1f}%，一般", 5
                            else:
                                return f"净利率 {margin:.1f}%，偏低", 3
        return "净利率数据不足", 5
    except Exception:
        return "净利率数据不足", 5


def _score_roe(calc_index_json: str) -> tuple[str, int]:
    """ROE 评分 (权重20%)"""
    try:
        data = json.loads(calc_index_json)
        items = data if isinstance(data, list) else data.get("items", data.get("data", [data]))
        for item in items if isinstance(items, list) else [items]:
            if isinstance(item, dict):
                for key in ("roe", "return_on_equity"):
                    if key in item:
                        roe = _safe_float(item[key])
                        if roe > 20:
                            return f"ROE {roe:.1f}%，优秀", 10
                        elif roe > 15:
                            return f"ROE {roe:.1f}%，良好", 8
                        elif roe > 10:
                            return f"ROE {roe:.1f}%，中等", 6
                        elif roe > 5:
                            return f"ROE {roe:.1f}%，偏低", 4
                        else:
                            return f"ROE {roe:.1f}%，较差", 2
        return "ROE 数据不足", 5
    except Exception:
        return "ROE 数据不足", 5


def _score_debt_ratio(financial_json: str) -> tuple[str, int]:
    """资产负债率评分 (权重15%)"""
    try:
        data = json.loads(financial_json)
        items = data if isinstance(data, list) else data.get("items", data.get("data", [data]))
        for item in (items if isinstance(items, list) else [items]):
            if isinstance(item, dict):
                for key in ("debt_to_assets", "debt_ratio", "资产负债率", "total_liab"):
                    if key in item:
                        ratio = _safe_float(item[key])
                        if "liab" in key and "assets" not in key:
                            # 可能是绝对值，跳过
                            continue
                        if ratio > 0:
                            if ratio < 30:
                                return f"资产负债率 {ratio:.1f}%，非常健康", 10
                            elif ratio < 50:
                                return f"资产负债率 {ratio:.1f}%，健康", 8
                            elif ratio < 70:
                                return f"资产负债率 {ratio:.1f}%，中等", 5
                            else:
                                return f"资产负债率 {ratio:.1f}%，偏高", 2
        return "负债率数据不足", 5
    except Exception:
        return "负债率数据不足", 5


def _score_cashflow(financial_json: str) -> tuple[str, int]:
    """现金流质量评分 (权重15%)"""
    try:
        data = json.loads(financial_json)
        items = data if isinstance(data, list) else data.get("items", data.get("data", [data]))
        for item in (items if isinstance(items, list) else [items]):
            if isinstance(item, dict):
                ocf = None
                np_ = None
                for key in ("operating_cash_flow", "cash_from_operations", "经营活动现金流净额"):
                    if key in item:
                        ocf = _safe_float(item[key])
                for key in ("net_income", "net_profit", "净利润"):
                    if key in item:
                        np_ = _safe_float(item[key])
                if ocf is not None and np_ and np_ > 0:
                    ratio = ocf / np_
                    if ratio > 1.2:
                        return f"经营现金流/净利润 {ratio:.1f}，优质", 10
                    elif ratio > 1.0:
                        return f"经营现金流/净利润 {ratio:.1f}，良好", 8
                    elif ratio > 0.7:
                        return f"经营现金流/净利润 {ratio:.1f}，一般", 5
                    else:
                        return f"经营现金流/净利润 {ratio:.1f}，偏弱", 3
        return "现金流数据不足", 5
    except Exception:
        return "现金流数据不足", 5


def _score_valuation(valuation_json: str) -> tuple[str, int]:
    """估值水平评分 (权重20%)"""
    try:
        data = json.loads(valuation_json)
        # 尝试找历史分位
        items = data if isinstance(data, list) else data.get("items", data.get("data", [data]))

        for item in (items if isinstance(items, list) else [items]):
            if isinstance(item, dict):
                # 找 PE 分位
                for key in ("pe_percentile", "pe_percent", "历史分位"):
                    if key in item:
                        pct = _safe_float(item[key])
                        if pct < 20:
                            return f"PE 处于历史 {pct:.0f}% 分位，极低估值", 10
                        elif pct < 40:
                            return f"PE 处于历史 {pct:.0f}% 分位，低估值", 8
                        elif pct < 60:
                            return f"PE 处于历史 {pct:.0f}% 分位，合理", 5
                        elif pct < 80:
                            return f"PE 处于历史 {pct:.0f}% 分位，偏高", 3
                        else:
                            return f"PE 处于历史 {pct:.0f}% 分位，高估", 1

                # 找 PE 值本身
                for key in ("pe", "pe_ttm", "市盈率"):
                    if key in item:
                        pe = _safe_float(item[key])
                        if pe < 0:
                            return f"PE {pe:.1f}（亏损），无法估值", 2
                        elif pe < 10:
                            return f"PE {pe:.1f}，低估值", 8
                        elif pe < 20:
                            return f"PE {pe:.1f}，合理估值", 6
                        elif pe < 40:
                            return f"PE {pe:.1f}，中等估值", 4
                        else:
                            return f"PE {pe:.1f}，高估值", 2

        # 尝试从原始 JSON 文本中提取分位信息
        text = valuation_json if isinstance(valuation_json, str) else json.dumps(data)
        for keyword in ("分位", "percentile"):
            if keyword in text.lower():
                import re
                match = re.search(r'(\d+(?:\.\d+)?)\s*%', text)
                if match:
                    pct = float(match.group(1))
                    return f"估值约处于历史 {pct:.0f}% 分位", max(1, 10 - int(pct / 10))

        return "估值数据不足", 5
    except Exception:
        return "估值数据不足", 5


def _score_dividend(dividend_json: str) -> str:
    """分红信息（不计入评分，仅展示）"""
    try:
        data = json.loads(dividend_json)
        items = data if isinstance(data, list) else data.get("items", data.get("data", [data]))
        if isinstance(items, list) and items:
            latest = items[0] if isinstance(items[0], dict) else {}
            yield_ = latest.get("yield", latest.get("dividend_yield", latest.get("股息率")))
            if yield_ is not None:
                return f"股息率 {_safe_float(yield_):.2f}%"
        return "无分红数据"
    except Exception:
        return "无分红数据"


def analyze_fundamentals(
    financial_report_json: str = "",
    valuation_json: str = "",
    consensus_json: str = "",
    dividend_json: str = "",
    calc_index_json: str = "",
) -> FundamentalResult:
    """综合基本面分析

    接收各 CLI 工具返回的 JSON 字符串，输出结构化评分。
    """
    # 各维度评分
    rev_desc, rev_score = _score_revenue_growth(financial_report_json)
    margin_desc, margin_score = _score_profit_margin(calc_index_json, financial_report_json)
    roe_desc, roe_score = _score_roe(calc_index_json)
    debt_desc, debt_score = _score_debt_ratio(financial_report_json)
    cf_desc, cf_score = _score_cashflow(financial_report_json)
    val_desc, val_score = _score_valuation(valuation_json)
    div_info = _score_dividend(dividend_json)

    # 加权总分
    weighted = (
        rev_score * 0.15 +
        margin_score * 0.15 +
        roe_score * 0.20 +
        debt_score * 0.15 +
        cf_score * 0.15 +
        val_score * 0.20
    )
    score = max(1, min(10, round(weighted)))

    if score >= 8:
        rating = "excellent"
    elif score >= 6:
        rating = "good"
    elif score >= 4:
        rating = "fair"
    else:
        rating = "poor"

    summary = f"基本面评分 {score}/10 ({rating}) | ROE:{roe_desc.split('，')[0] if '，' in roe_desc else roe_desc} | {val_desc.split('，')[0] if '，' in val_desc else val_desc}"

    return FundamentalResult(
        revenue_growth=rev_desc,
        profit_margin=margin_desc,
        roe=roe_desc,
        debt_ratio=debt_desc,
        cashflow_quality=cf_desc,
        valuation_level=val_desc,
        dividend_info=div_info,
        score=score,
        rating=rating,
        summary=summary,
    )


def format_fundamental_report(result: FundamentalResult, symbol: str) -> str:
    """格式化基本面分析报告为 Markdown"""
    rating_cn = {"excellent": "优秀", "good": "良好", "fair": "一般", "poor": "较差"}
    lines = [
        f"## 基本面分析: {symbol}",
        f"",
        f"| 维度 | 评估 |",
        f"|------|------|",
        f"| 营收增长 | {result['revenue_growth']} |",
        f"| 净利率 | {result['profit_margin']} |",
        f"| ROE | {result['roe']} |",
        f"| 负债率 | {result['debt_ratio']} |",
        f"| 现金流 | {result['cashflow_quality']} |",
        f"| 估值水平 | {result['valuation_level']} |",
        f"| 分红 | {result['dividend_info']} |",
        f"| **综合评分** | **{result['score']}/10 ({rating_cn.get(result['rating'], result['rating'])})** |",
        f"",
        f"> {result['summary']}",
    ]
    return "\n".join(lines)
