"""交易记忆日志

追加式日志，记录每次分析决策，支持跨会话反思学习。
文件路径: data/journal.jsonl
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).parent.parent.parent / "data"
JOURNAL_FILE = DATA_DIR / "journal.jsonl"


def append_decision(
    symbol: str,
    action: str,
    reasoning: str = "",
    entry_price: Optional[float] = None,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    confidence: str = "medium",
    user_preference: str = "",
):
    """追加一条决策记录"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "symbol": symbol,
        "action": action,             # buy_recommend / sell_recommend / hold / wait
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "confidence": confidence,
        "reasoning": reasoning[:500],  # 限制长度
        "user_preference": user_preference,
    }

    with open(JOURNAL_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def get_recent_decisions(n: int = 10) -> list[dict]:
    """获取最近 N 条决策"""
    if not JOURNAL_FILE.exists():
        return []
    lines = JOURNAL_FILE.read_text(encoding="utf-8").strip().split("\n")
    decisions = []
    for line in lines[-n:]:
        line = line.strip()
        if line:
            try:
                decisions.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return decisions


def get_symbol_history(symbol: str) -> list[dict]:
    """获取某股票的历史决策"""
    if not JOURNAL_FILE.exists():
        return []
    decisions = []
    for line in JOURNAL_FILE.read_text(encoding="utf-8").strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            if entry.get("symbol", "").upper() == symbol.upper():
                decisions.append(entry)
        except json.JSONDecodeError:
            continue
    return decisions


def get_reflection_context(n: int = 5) -> str:
    """生成反思上下文文本，注入 Agent prompt"""
    recent = get_recent_decisions(n)
    if not recent:
        return ""

    lines = ["## 近期决策记录（供参考）"]
    for d in recent[-n:]:
        ts = d.get("timestamp", "")[:16]
        sym = d.get("symbol", "?")
        action = d.get("action", "?")
        conf = d.get("confidence", "?")
        reason = d.get("reasoning", "")[:100]
        entry = d.get("entry_price", "")
        sl = d.get("stop_loss", "")

        action_cn = {"buy_recommend": "建议买入", "sell_recommend": "建议卖出",
                     "hold": "持有", "wait": "观望"}.get(action, action)

        line = f"- [{ts}] {sym} {action_cn}(信心:{conf})"
        if entry:
            line += f" 入场:{entry}"
        if sl:
            line += f" 止损:{sl}"
        if reason:
            line += f" | {reason}"
        lines.append(line)

    return "\n".join(lines)
