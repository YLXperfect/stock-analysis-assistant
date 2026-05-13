"""5-Agent 编排系统

Router (纯意图识别，无工具)
  ├→ Query Agent (数据查询和展示) → END
  ├→ Analyst (数据收集 + 多维分析) → Strategist (多空辩论 + 交易策略) → END
  ├→ Strategist (直接策略建议) → END
  └→ Screener (条件选股) → END

多轮对话: LangGraph InMemorySaver checkpointer 持久化会话状态
上下文管理: Token 数超过 50K 时自动 LLM 摘要压缩
"""

import json
from typing import Annotated
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from src.agent.llm import create_llm
from src.tools.longbridge_sdk import get_analyst_tools
from src.tools.longbridge_sdk import analyze_technical
from src.tools.longbridge_sdk import get_positions, get_portfolio, get_assets
from src.tools.longbridge_sdk import get_watchlist_tools, run_backtest
from src.tools.longbridge_sdk import (
    watchlist_list, watchlist_show,
    get_calc_index, get_financial_report, get_valuation,
)
from src.tools.analysis_tools import (
    analyze_fundamental,
    analyze_strategy, analyze_dca,
    analyze_position_safety, analyze_portfolio,
    quick_screen_stock, get_stock_universe,
    analyze_backtest,
)
from src.tools.profile_tools import get_user_profile, update_user_profile
from src.memory.profile import get_profile_summary
from src.memory.journal import append_decision, get_reflection_context
from src.skills import build_skill_prompt

# 加载分析方法论 Skill
_ANALYSIS_SKILL = ""
try:
    _ANALYSIS_SKILL = build_skill_prompt("analysis")
except Exception:
    pass


# ── Agent Prompts ───────────────────────────────────────────────

ROUTER_PROMPT = """你是意图分类器。根据用户消息，判断应该路由到哪个 Agent。
只回复路由指令词，不回答问题、不解释。

## 路由规则

回复 ROUTE_QUERY 当用户要:
- 查看/显示/查询行情、资金流向、盘口、分时、K线
- 查看新闻、财报、估值、分红、评级、内幕交易
- 管理自选股、价格提醒
- 查看持仓、账户资产
- 管理偏好设置
- 扫描自选股

回复 ROUTE_ANALYSIS 当用户要:
- 分析/评估/诊断/研究某只股票

回复 ROUTE_STRATEGY 当用户要:
- 买/卖/入场/止损/止盈/策略/定投/回测
- 该不该买、能不能买、值不值得买
- 持仓安全评估

回复 ROUTE_SCREENING 当用户要:
- 筛选/选股/过滤/找股票（带条件）

回复 ROUTE_UNSUPPORTED 当:
- 交易下单、宏观经济、行业研报、期权、加密货币等不支持的功能

## 股票名称映射
- 腾讯 → 700.HK, 阿里 → 9988.HK, 美团 → 3690.HK
- 苹果 → AAPL.US, 特斯拉 → TSLA.US, 英伟达 → NVDA.US

只回复一个路由指令词，不要多余文字。"""


QUERY_PROMPT = """你是股票数据查询助手。根据用户需求，调用合适的工具获取数据并以清晰格式展示。

## 你的工具（按场景选择）
- 行情: get_quote, get_depth, get_intraday, get_calc_index
- K线: get_kline
- 资金面: get_capital_flow
- 基本面: get_static_info, get_financial_report, get_valuation, get_dividend, get_company_overview
- 评级/内幕: get_institution_rating, get_insider_trades
- 新闻: get_news, search_news
- 账户: get_positions, get_portfolio, get_assets
- 自选股: watchlist_list, watchlist_show, watchlist_create, watchlist_update, watchlist_delete, watchlist_pin
- 提醒: alert_list, alert_add, alert_delete, alert_enable, alert_disable
- 扫描: scan_watchlist
- 偏好: get_user_profile, update_user_profile

## 股票名称映射
- 腾讯/腾讯控股 → 700.HK, 阿里/阿里巴巴 → 9988.HK
- 美团 → 3690.HK, 小米 → 1810.HK, 比亚迪 → 1211.HK
- 苹果 → AAPL.US, 特斯拉 → TSLA.US, 英伟达 → NVDA.US
- 茅台/贵州茅台 → 600519.SH, 宁德时代 → 300750.SZ

## 规则
- 只做数据查询和展示，不做分析、不给买卖建议
- 数据以表格或列表形式展示，让用户一目了然
- 如果数据获取失败，如实告知原因
- 用简洁中文回答。如果用户用英文提问，用英文回答
"""

ANALYST_PROMPT = """你是专业的股票分析师。根据用户需求，系统性地收集数据并输出结构化分析报告。

## 分析流程
1. get_quote → 实时行情概况
2. analyze_technical → 技术面评分（MA/MACD/RSI/布林带）
3. analyze_fundamental → 深度基本面评分（营收/利润/ROE/负债/现金流/估值）
4. get_capital_flow → 资金面（主力资金流入流出）
5. get_news → 近期新闻/催化事件
6. get_institution_rating → 机构评级和目标价
7. get_insider_trades → 内幕交易动向（美股）

根据用户的具体问题选择性调用，不需要每次都调全部。

## 你的工具
### 行情类
- get_quote(symbol): 实时行情
- get_kline(symbol, period, count, adjust): K线数据
- get_depth(symbol): 盘口深度
- get_intraday(symbol): 分时数据
- get_capital_flow(symbol): 资金流向
- get_calc_index(symbol, fields): 计算指标

### 基本面类
- get_financial_report(symbol, kind, report): 财务报表
- get_valuation(symbol): 估值分析
- get_dividend(symbol): 分红历史
- get_forecast_eps(symbol): EPS 预测
- get_consensus(symbol): 财务共识
- get_institution_rating(symbol): 机构评级
- get_company_overview(symbol): 公司概况
- get_shareholders(symbol): 机构股东

### 新闻类
- get_news(symbol, count): 新闻
- search_news(keyword): 搜索新闻
- get_insider_trades(symbol): 内幕交易

### 分析类
- analyze_technical(symbol): 技术面综合分析
- analyze_fundamental(symbol): 深度基本面分析

## 输出格式

**当请求同时包含查询和分析时，分两部分输出：**

1. **数据展示**（先输出）：用表格/列表展示用户要求查看的数据（行情、资金流向、新闻等），让用户快速看到原始数据
2. **分析报告**（后输出）：每个维度单独评分 + 综合评分，给出客观分析

**纯查询请求也由你处理**——直接展示数据即可，不需要分析。
**纯分析请求**——按分析流程执行。

**不要给出买卖建议**——那是策略师的工作。你只负责客观分析。

## 能力边界
- 你只做**个股分析**（技术面/基本面/资金面/新闻）
- 不做行业分析、宏观经济、市场情绪判断
- 如果数据不足导致无法分析，直接说明"数据不足"，不要编造
- 如果用户用英文提问，用英文回答

## 分析工作流 Skill
""" + _ANALYSIS_SKILL + """
"""

STRATEGIST_PROMPT = """你是高级投资策略师。基于分析数据，给出结构化的交易建议。

## 你的核心方法 — 多空辩论
对每只股票，你都要:
1. 列出所有**看多论据**（利好信号）
2. 列出所有**看空论据**（风险因素）
3. 综合评判，给出倾向性结论

## 你的工具
- analyze_technical(symbol): 技术面分析 + 评分
- analyze_fundamental(symbol): 基本面分析 + 评分
- analyze_strategy(symbol, strategy_type): 策略分析 + 多空辩论 + 交易提案
- analyze_dca(symbol, budget, intervals): 定投策略
- analyze_position_safety(symbol): 持仓安全评估
- analyze_portfolio(): 组合全面分析
- analyze_backtest(symbol, start, end, strategy_type, **params): 策略回测
- run_backtest(symbol, start, end, script, period): 自定义回测
- get_positions(): 获取当前持仓
- get_portfolio(): 获取投资组合
- get_user_profile(): 获取用户风险偏好

## 策略回测
当用户要求回测时，优先使用 analyze_backtest 高级工具（内置策略模板）:
- analyze_backtest(symbol, start, end, strategy_type="ma_cross")
- analyze_backtest(symbol, start, end, strategy_type="rsi")
- analyze_backtest(symbol, start, end, strategy_type="bollinger")
- analyze_backtest(symbol, start, end, strategy_type="macd")

如果用户描述的策略不在模板中，用 run_backtest 自定义 PineScript。

### 可用策略类型
- ma_cross: 均线交叉（参数: fast_period, slow_period）
- rsi: RSI 超买超卖（参数: period, oversold, overbought）
- bollinger: 布林带突破（参数: period, stddev）
- macd: MACD 金叉死叉

### 回测结果解读
输出夏普比率、最大回撤、胜率、利润因子等指标。
结合回测结果给出交易建议：回测表现好的策略，信心度更高。

## 输出语言规范（严格遵守）
- 所有数据都是你通过工具查询得到的，**严禁**说"根据您提供的数据/根据您给的股价/根据您输入的价格"
- 正确说法："根据查询，XX股价为XXX" 或 "通过工具获取的数据显示"
- 如果数据由上游分析师提供，说"根据分析报告"即可

## 输出格式 — 结构化交易提案

操作: 买入/持有/卖出/观望
信心度: 高/中/低
入场价: xxx-xxx (说明)
止损价: xxx (-x%)
止盈价: xxx (+x%)
风险收益比: 1:x.x
建议仓位: 总资金的 x%

看多论据:
- ...

看空论据:
- ...

综合理由: ...

## 必须参考
1. 调用 get_user_profile() 获取用户风险偏好
2. 仓位建议基于用户的 max_position_pct
3. 止损建议基于用户的 stop_loss_default_pct
4. 如果用户问持仓安全 → 调用 analyze_portfolio 或 analyze_position_safety

## 能力边界
- 你只处理与**个股交易决策**和**持仓管理**相关的问题
- 回测仅支持 4 种预设策略：ma_cross/rsi/bollinger/macd
- 不做宏观判断、不做板块推荐、不推荐具体股票代码
- 如果用户问的问题不属于策略/交易范畴，诚实说明应由哪个 Agent 处理
- 如果用户用英文提问，用英文回答

{user_profile}
{reflection_context}
"""

SCREENING_PROMPT = """你是股票筛选助手。根据用户的筛选条件，从股票池中找到符合条件的股票。

## 筛选流程
1. 确定股票池来源：
   - 用户说 "我的自选股" → 调用 watchlist_show("all")
   - 用户说 "美股/港股" → 调用 get_stock_universe("US" 或 "HK")
   - 用户指定具体股票 → 直接使用

2. 对池中每只股票调用 quick_screen_stock(symbol) 获取核心指标
   - 返回 PE、PB、市值、ROE、营收增长、股息率等

3. 根据用户条件筛选：
   - 估值条件：PE、PB、PS
   - 成长条件：营收增长、利润增长
   - 质量条件：ROE、负债率
   - 规模条件：市值
   - 收益条件：股息率

4. 对通过初步筛选的股票，可调用 get_valuation 或 get_financial_report 做深入分析

5. 输出格式 — 按匹配度排序的表格:
   | 代码 | 名称 | PE | ROE | 市值 | 营收增长 | 股息率 | 评级 |
   每只标注达标/未达标条件。

## 你的工具
- quick_screen_stock(symbol): 快速获取单只股票筛选指标（优先使用）
- get_stock_universe(market): 获取预定义股票池（"US" 或 "HK"）
- watchlist_show(group): 从自选股获取股票列表
- watchlist_list(): 列出所有自选股分组
- get_calc_index(symbol, fields): 获取补充指标
- get_financial_report(symbol, kind, report): 获取详细财务数据
- get_valuation(symbol): 估值分析

## 重要提示
- 批量筛选时，优先使用 quick_screen_stock 做初步过滤
- 只对初步通过的股票做深入分析，避免对每只股票都调大量工具
- 如果股票池很大（>30只），分批处理，先取前30只
- 用简洁的中文回答，结果用表格展示

## 能力边界
- 股票池仅限：美股 ~50 只 + 港股 ~30 只（通过 get_stock_universe 获取）
- 如果用户指定的股票不在池中，用 get_quote 查一下，告诉用户 "该股票暂不在选股池中"
- 不提供其他市场（A 股/新加坡/日本等）的选股
- 筛选维度限制：PE/PB/市值/ROE/营收增长/股息率
- 如果用户用英文提问，用英文回答
"""


# ── State ───────────────────────────────────────────────────────

class MultiAgentState(BaseModel):
    query: str = ""
    messages: Annotated[list, add_messages] = []
    route_decision: str = ""     # simple / analysis / strategy / screening
    analysis_report: str = ""    # Analyst 的输出
    conversation_summary: str = ""  # 旧对话的 LLM 摘要（token 超限时压缩）


# ── 上下文管理 ────────────────────────────────────────────────

def _estimate_tokens(messages: list) -> int:
    """粗略估算 token 数（中英文平均 1 token ≈ 2 字符）"""
    total = 0
    for m in messages:
        content = m.content if hasattr(m, "content") else str(m)
        total += len(content)
    return total // 2


def _build_context(state: "MultiAgentState") -> str:
    """构建对话上下文前缀，注入到各 Agent 的第一次用户消息中"""
    parts = []

    # 旧对话摘要
    if state.conversation_summary:
        parts.append(f"## 此前对话摘要\n{state.conversation_summary}")

    # 最近消息（排除当前 query 的 AIMessage 和 HumanMessage 重复）
    prior_msgs = []
    for m in state.messages:
        if isinstance(m, HumanMessage) and m.content == state.query:
            break  # 到当前轮为止
        prior_msgs.append(m)

    if prior_msgs:
        parts.append("## 最近对话")
        for m in prior_msgs[-8:]:  # 最多显示最近 4 轮
            role = "用户" if isinstance(m, HumanMessage) else "助手"
            content = m.content if hasattr(m, "content") else str(m)
            parts.append(f"{role}: {content[:300]}")

    if parts:
        parts.append("\n---\n")
    return "\n".join(parts)


async def _maybe_summarize(state: "MultiAgentState", llm) -> dict:
    """当消息 token 数超过阈值时，压缩旧消息为摘要"""
    msgs = state.messages
    if _estimate_tokens(msgs) < 50_000:
        return {}

    # 取前 2/3 的消息用 LLM 压缩
    split = max(1, len(msgs) * 2 // 3)
    old = msgs[:split]
    recent = msgs[split:]

    summary_prompt = (
        "将以下对话压缩为要点摘要，只保留关键信息："
        "涉及的股票代码、操作建议、价格数据、用户偏好。"
        "限制 300 字以内。\n\n"
        + "\n".join(
            f"{'用户' if isinstance(m, HumanMessage) else '助手'}: {m.content[:200]}"
            for m in old[-30:]  # 最多取最近 30 条旧消息
        )
    )
    try:
        resp = await llm.ainvoke(summary_prompt)
        new_summary = resp.content[:500]
    except Exception:
        new_summary = "（历史对话摘要）"

    # 合并已有摘要
    merged = new_summary
    if state.conversation_summary:
        merged = state.conversation_summary + "\n" + new_summary

    return {
        "messages": list(recent),
        "conversation_summary": merged[:800],  # 摘要总长度限制
    }


# ── Agent Nodes ─────────────────────────────────────────────────

def build_multi_agent(llm=None):
    """构建 4-Agent 编排图

    Args:
        llm: LangChain LLM 实例，None 则自动创建

    Returns:
        编译后的 LangGraph CompiledGraph
    """
    if llm is None:
        llm = create_llm()

    from langchain.agents import create_agent
    from langchain_core.messages import HumanMessage as _HM, AIMessage as _AIM

    # ── 准备工具 ───────────────────────────
    # Query Agent 工具：所有数据查询和账户管理
    from src.tools.longbridge_sdk import (
        get_quote, get_static_info, get_calc_index,
        get_kline, get_depth, get_intraday, get_capital_flow,
        get_financial_report, get_valuation, get_dividend,
        get_institution_rating, get_company_overview,
        get_news, search_news, get_insider_trades,
        get_positions, get_portfolio, get_assets,
        watchlist_list, watchlist_show, watchlist_create,
        watchlist_update, watchlist_delete, watchlist_pin,
        alert_list, alert_add, alert_delete,
        alert_enable, alert_disable,
        scan_watchlist,
    )

    query_tools = [
        get_quote, get_static_info, get_calc_index,
        get_kline, get_depth, get_intraday, get_capital_flow,
        get_financial_report, get_valuation, get_dividend,
        get_institution_rating, get_company_overview,
        get_news, search_news, get_insider_trades,
        get_positions, get_portfolio, get_assets,
        watchlist_list, watchlist_show, watchlist_create,
        watchlist_update, watchlist_delete, watchlist_pin,
        alert_list, alert_add, alert_delete,
        alert_enable, alert_disable,
        scan_watchlist,
        get_user_profile, update_user_profile,
    ]

    analyst_tools = get_analyst_tools() + [analyze_technical, analyze_fundamental]

    strategist_tools = [
        analyze_technical, analyze_fundamental,
        analyze_strategy, analyze_dca,
        analyze_position_safety, analyze_portfolio,
        analyze_backtest, run_backtest,
        get_user_profile,
        get_positions, get_portfolio, get_assets,
    ]
    strategist_tools += get_watchlist_tools()

    screening_tools = [
        quick_screen_stock, get_stock_universe,
        watchlist_list, watchlist_show,
        get_calc_index, get_financial_report,
        get_valuation,
    ]

    # ── 构建子 Agent ───────────────────────
    query_agent = create_agent(model=llm, tools=query_tools, system_prompt=QUERY_PROMPT)
    analyst_agent = create_agent(model=llm, tools=analyst_tools, system_prompt=ANALYST_PROMPT)

    profile_summary = ""
    try:
        profile_summary = get_profile_summary()
    except Exception:
        pass

    reflection = ""
    try:
        reflection = get_reflection_context(5)
    except Exception:
        pass

    strategist_prompt = STRATEGIST_PROMPT.format(
        user_profile=f"\n## 当前用户偏好\n{profile_summary}" if profile_summary else "",
        reflection_context=f"\n## 近期决策记忆\n{reflection}" if reflection else "",
    )
    strategist_agent = create_agent(model=llm, tools=strategist_tools, system_prompt=strategist_prompt)
    screening_agent = create_agent(model=llm, tools=screening_tools, system_prompt=SCREENING_PROMPT)

    # ── Graph Nodes ───────────────────────

    async def router_node(state: MultiAgentState) -> dict:
        """Router: 纯意图识别（关键词优先 + LLM 兜底），不回答问题"""
        import re as _re
        query = state.query

        # 追问消歧：如果当前 query 没有股票代码，从上一轮对话中提取
        _has_symbol = _re.search(r'[A-Z]{2,5}\.[A-Z]{2,3}|\d{4,6}\.[A-Z]{2,3}', query)
        if not _has_symbol:
            for msg in reversed(state.messages[-4:]):
                content = msg.content if hasattr(msg, "content") else ""
                syms = _re.findall(r'\b([A-Z]{2,5}\.\w{2,3}|\d{4,6}\.\w{2,3})\b', content)
                if syms:
                    query = f"{syms[-1]} {query}"
                    break

        # ── 1. 关键词匹配（优先） ──────────
        # 不支持的功能 → 直接返回拒绝消息
        unsupported_patterns = [
            r'(挂单|下单).*\d+\s*(股|手)',
            r'(买入|卖出)\s*\d+\s*(股|手)',
            r'以?\s*[\d.]+\s*(元|美元|港币)?\s*(买入|卖出)',
            r'(市价|限价)\s*(买入|卖出)',
        ]
        unsupported_kw = ["GDP", "CPI", "美联储", "非农", "加息", "期权", "Call", "Put",
                          "BTC", "ETH", "比特币", "加密货币", "期货", "外汇", "打新", "IPO"]
        if any(_re.search(p, query) for p in unsupported_patterns) or \
           any(kw in query for kw in unsupported_kw):
            return {
                "messages": [_AIM(content="抱歉，当前暂不支持该功能。目前支持：个股行情查询、技术/基本面分析、交易策略建议、条件选股、自选股和价格提醒管理。")],
                "route_decision": "unsupported",
            }

        # 意图分类关键词
        has_analysis = any(kw in query for kw in ["分析", "评估", "诊断", "研究"])
        has_strategy = any(kw in query for kw in ["买", "卖", "入场", "止损", "止盈", "策略",
                                                     "定投", "该不该", "能不能买", "值得买", "回测"])
        has_screening = any(kw in query for kw in ["筛选", "选股", "过滤", "找股票"])

        force_route = ""
        if has_screening:
            force_route = "screening"
        elif has_strategy:
            force_route = "strategy"
        elif has_analysis:
            force_route = "analysis"

        if force_route:
            return {"messages": [], "route_decision": force_route}

        # ── 2. LLM 兜底（纯文本分类，无工具） ──
        route = "query"  # 默认走 Query Agent
        try:
            llm_result = await llm.ainvoke(ROUTER_PROMPT + "\n\n用户消息: " + query)
            route_text: str = getattr(llm_result, "content", "ROUTE_QUERY").strip().upper()
            if "ROUTE_ANALYSIS" in route_text:
                route = "analysis"
            elif "ROUTE_STRATEGY" in route_text:
                route = "strategy"
            elif "ROUTE_SCREENING" in route_text:
                route = "screening"
            elif "ROUTE_UNSUPPORTED" in route_text:
                return {
                    "messages": [_AIM(content="抱歉，当前暂不支持该功能。目前支持：个股行情查询、技术/基本面分析、交易策略建议、条件选股、自选股和价格提醒管理。")],
                    "route_decision": "unsupported",
                }
        except Exception:
            pass  # LLM 分类失败时默认走 query

        return {"messages": [], "route_decision": route}

    async def query_node(state: MultiAgentState) -> dict:
        """Query Agent: 数据查询和展示"""
        context = _build_context(state)
        result = await query_agent.ainvoke(
            {"messages": [_HM(content=context + state.query)]}
        )
        final_msg = result["messages"][-1] if result.get("messages") else None
        return {"messages": [AIMessage(content=final_msg.content if final_msg else "查询完成")]}

    async def analyst_node(state: MultiAgentState) -> dict:
        """Analyst: 数据收集 + 多维分析"""
        context = _build_context(state)
        query = context + state.query

        result = await analyst_agent.ainvoke(
            {"messages": [_HM(content= query)]}
        )
        final_msg = result["messages"][-1] if result.get("messages") else None
        analysis_text = final_msg.content if final_msg else "分析完成"

        return {
            "messages": [],
            "analysis_report": analysis_text,
        }

    async def strategist_node(state: MultiAgentState) -> dict:
        """Strategist: 策略 + 多空辩论 + 交易提案"""
        context = _build_context(state)
        if state.analysis_report:
            strategist_input = (
                f"{context}用户问题: {state.query}\n\n"
                f"以下是由分析师提供的分析报告:\n{state.analysis_report}\n\n"
                f"请基于以上分析，给出你的策略建议和多空辩论。"
            )
        else:
            strategist_input = context + state.query

        result = await strategist_agent.ainvoke(
            {"messages": [_HM(content= strategist_input)]}
        )
        final_msg = result["messages"][-1] if result.get("messages") else None
        response_text = final_msg.content if final_msg else "策略分析完成"

        # 写入交易记忆
        try:
            _extract_and_log_decision(state.query, response_text)
        except Exception:
            pass

        return {"messages": [AIMessage(content=response_text)]}

    async def screening_node(state: MultiAgentState) -> dict:
        """Screener: 条件选股 + 批量筛选"""
        context = _build_context(state)
        result = await screening_agent.ainvoke(
            {"messages": [_HM(content= context + state.query)]}
        )
        final_msg = result["messages"][-1] if result.get("messages") else None
        response_text = final_msg.content if final_msg else "筛选完成"

        return {"messages": [AIMessage(content=response_text)]}

    # ── 路由函数 ───────────────────────────

    def route_after_router(state: MultiAgentState) -> str:
        return state.route_decision  # query / analysis / strategy / screening / unsupported

    # ── 构建图 ─────────────────────────────

    graph = StateGraph(MultiAgentState)

    graph.add_node("router", router_node)
    graph.add_node("query", query_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("strategist", strategist_node)
    graph.add_node("screener", screening_node)

    graph.add_edge(START, "router")

    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "query": "query",
            "analysis": "analyst",
            "strategy": "strategist",
            "screening": "screener",
            "unsupported": END,
        },
    )

    graph.add_edge("query", END)
    graph.add_edge("analyst", "strategist")
    graph.add_edge("strategist", END)
    graph.add_edge("screener", END)

    return graph.compile(checkpointer=InMemorySaver())


def _extract_and_log_decision(query: str, response: str):
    """从策略输出中提取决策并写入日志"""
    import re

    # 尝试提取股票代码
    symbol = ""
    symbol_match = re.search(r'(\d{4,6}\.[A-Z]{2}|[A-Z]{1,5}\.US)', query + response)
    if symbol_match:
        symbol = symbol_match.group(1)

    if not symbol:
        return

    # 判断行动
    action = "hold"
    if "买入" in response or "建议买入" in response:
        action = "buy_recommend"
    elif "卖出" in response or "建议卖出" in response:
        action = "sell_recommend"
    elif "观望" in response or "等待" in response:
        action = "wait"

    # 提取价格
    entry_price = None
    sl = None
    tp = None
    confidence = "medium"

    for label, var_name in [("入场价", "entry"), ("止损价", "sl"), ("止盈价", "tp")]:
        match = re.search(rf'{label}[^0-9]*([0-9]+\.?[0-9]*)', response)
        if match:
            if var_name == "entry":
                entry_price = float(match.group(1))
            elif var_name == "sl":
                sl = float(match.group(1))
            elif var_name == "tp":
                tp = float(match.group(1))

    if "信心度: 高" in response or "信心度高" in response:
        confidence = "high"
    elif "信心度: 低" in response or "信心度低" in response:
        confidence = "low"

    # 提取理由（取前 500 字）
    reasoning = response[:500]

    append_decision(
        symbol=symbol,
        action=action,
        reasoning=reasoning,
        entry_price=entry_price,
        stop_loss=sl,
        take_profit=tp,
        confidence=confidence,
    )
