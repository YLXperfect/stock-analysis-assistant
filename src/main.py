"""Stock Analysis Assistant - 主入口

用法:
    python -m src.main
    python -m src.main -q "分析一下腾讯(700.HK)"

数据源: Longbridge Python SDK
框架: LangChain + LangGraph (Multi-Agent)
"""

import asyncio
import argparse
import uuid
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.live import Live
from rich.text import Text

from typing import Any
from config.settings import settings
from src.agent.llm import create_llm
from src.agent.multi_agent import build_multi_agent, MultiAgentState

console = Console()

# ── 工具/节点名称中文映射 ─────────────────────────────

TOOL_DISPLAY: dict[str, str] = {
    "get_quote": "查询实时行情",
    "get_kline": "获取K线数据",
    "get_capital_flow": "查询资金流向",
    "get_depth": "查询盘口深度",
    "get_intraday": "获取分时数据",
    "get_news": "获取新闻资讯",
    "get_financial_report": "获取财务报表",
    "get_valuation": "查询估值数据",
    "get_calc_index": "获取计算指标",
    "get_static_info": "获取公司信息",
    "get_dividend": "查询分红历史",
    "get_institution_rating": "查询机构评级",
    "get_insider_trades": "查询内幕交易",
    "get_company_overview": "获取公司概况",
    "search_news": "搜索新闻",
    "analyze_technical": "技术面分析",
    "analyze_fundamental": "基本面分析",
    "quick_screen_stock": "快速筛选",
    "run_backtest": "运行策略回测",
}

NODE_DISPLAY: dict[str, str] = {
    "query": "🔍 数据查询",
    "analyst": "📊 深度分析",
    "strategist": "🎯 策略生成",
    "screener": "🔎 条件选股",
}


# ── 流式输出核心 ──────────────────────────────────────

def _build_panel_content(node: str, tool_log: list[str], text: str, turn: int) -> Panel:
    """构建面板内容：进度提示 + 流式文本"""
    parts: list[str] = []

    # 进度提示（只有 analyst/strategist/screener 才显示）
    if node in ("analyst", "strategist", "screener"):
        node_label = NODE_DISPLAY.get(node, node)
        parts.append(f"[dim]{node_label}中...[/dim]")
        for i, t in enumerate(tool_log):
            prefix = "  ✓" if i < len(tool_log) - 1 else "  →"
            parts.append(f"[dim]{prefix} {t}[/dim]")
        parts.append("")

    # 流式文本
    if text:
        parts.append(text)

    content = "\n".join(parts) if parts else "[dim]思考中...[/dim]"
    return Panel(
        Markdown(content),
        title=f"[bold]分析报告[/bold] (第 {turn} 轮)",
    )


async def stream_graph_response(
    graph: Any,
    query: str,
    config: dict,
    turn: int,
) -> str:
    """流式输出 Agent 回复，带进度提示

    Returns:
        最终输出的完整文本
    """
    final_text = ""
    current_node = ""
    tool_log: list[str] = []

    with Live(console=console, refresh_per_second=8) as live:
        live.update(_build_panel_content(current_node, tool_log, final_text, turn))

        async for event in graph.astream_events(
            MultiAgentState(query=query),
            config=config,
            version="v2",
        ):
            kind = event["event"]
            name = event.get("name", "")

            # ── 节点开始 ──
            if kind == "on_chain_start" and name in NODE_DISPLAY:
                current_node = name
                tool_log = []  # 新节点重置工具日志
                live.update(_build_panel_content(current_node, tool_log, final_text, turn))

            # ── 工具调用 ──
            if kind == "on_tool_start":
                display = TOOL_DISPLAY.get(name, name)
                tool_log.append(display)
                live.update(_build_panel_content(current_node, tool_log, final_text, turn))

            # ── LLM token 流 ──
            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                token = getattr(chunk, "content", "")
                if token and current_node in ("query", "strategist", "screener"):
                    final_text += token
                    live.update(_build_panel_content(current_node, tool_log, final_text, turn))

    return final_text


# ── 入口函数 ──────────────────────────────────────────

async def run_analysis(query: str):
    """执行一次股票分析"""
    console.print(Panel(
        "[bold green]Stock Analysis Assistant[/bold green]\n"
        "Powered by Longbridge SDK + LangChain + LangGraph",
        title="Welcome",
    ))

    console.print("\n[bold]Step 1:[/bold] 初始化 LLM...")
    llm = create_llm()
    model_name = settings.llm_model if settings.use_openai_compat else settings.openai_model
    console.print(f"  模型: {model_name}")

    console.print("\n[bold]Step 2:[/bold] 构建 Agent...")
    graph = build_multi_agent(llm)
    console.print("[green]5-Agent 系统就绪![/green]")

    console.print(f"\n[bold]Step 3:[/bold] 分析中: [cyan]{query}[/cyan]")
    console.print("-" * 60)

    try:
        config: dict[str, Any] = {"configurable": {"thread_id": str(uuid.uuid4())}}
        final_text = await stream_graph_response(graph, query, config, turn=1)

        if not final_text:
            console.print("[red]分析失败: 未获得回复[/red]")
    except Exception as e:
        console.print(f"[red]分析出错: {e}[/red]")


async def interactive_mode():
    """交互模式 — 支持多轮对话记忆 + 流式输出"""
    thread_id = str(uuid.uuid4())
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}

    console.print(Panel(
        "[bold green]Stock Analysis Assistant[/bold green]\n"
        "输入股票代码或名称进行分析，输入 quit 退出\n"
        "支持多轮对话：可以追问、对比、深入分析\n"
        "示例: 700.HK, 腾讯, AAPL.US, 特斯拉\n\n"
        f"会话 ID: {thread_id[:8]}...\n"
        "数据源: Longbridge Python SDK",
        title="Welcome",
    ))

    llm = create_llm()
    graph = build_multi_agent(llm)
    console.print("[green]5-Agent 系统就绪![/green] (支持多轮对话记忆)\n")

    turn = 0
    while True:
        try:
            query = console.input("[bold cyan]You> [/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Bye![/yellow]")
            break

        if query.lower() in ("quit", "exit", "q"):
            console.print("[yellow]Bye![/yellow]")
            break

        if not query:
            continue

        console.print("-" * 60)
        turn += 1

        try:
            final_text = await stream_graph_response(graph, query, config, turn)

            if not final_text:
                console.print("[red]未获得回复[/red]")
        except Exception as e:
            console.print(f"[red]分析出错: {e}[/red]")

        console.print()


def main():
    parser = argparse.ArgumentParser(description="Stock Analysis Assistant")
    parser.add_argument("-q", "--query", type=str, help="直接分析，如 '分析一下腾讯(700.HK)'")
    parser.add_argument("-i", "--interactive", action="store_true", help="交互模式")
    args = parser.parse_args()

    if args.query:
        asyncio.run(run_analysis(args.query))
    else:
        asyncio.run(interactive_mode())


if __name__ == "__main__":
    main()
