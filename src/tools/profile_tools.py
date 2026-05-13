"""偏好管理工具 — Agent 1/3 共用

封装 src/memory/profile.py 为 LangChain @tool。
"""

from langchain_core.tools import tool


@tool
def get_user_profile() -> str:
    """获取用户的风险偏好和投资风格配置"""
    from src.memory.profile import get_profile_summary
    return f"## 当前用户偏好\n{get_profile_summary()}"


@tool
def update_user_profile(preference: str, value: str) -> str:
    """更新用户的投资偏好设置

    Args:
        preference: 偏好项，如 risk_tolerance, investment_style,
                    max_position_pct, stop_loss_default_pct, preferred_sectors
        value: 偏好值，如 aggressive, value, 15, 科技,医药
    """
    from src.memory.profile import update_profile, get_profile_summary

    # 值映射（中文 → 英文）
    value_map = {
        "保守": "conservative", "稳健": "moderate", "激进": "aggressive",
        "成长": "growth", "成长型": "growth",
        "价值": "value", "价值型": "value",
        "红利": "dividend", "红利型": "dividend",
        "均衡": "balanced", "均衡型": "balanced",
    }
    mapped_value = value_map.get(value, value)

    try:
        profile = update_profile(**{preference: mapped_value})
        return f"已更新 {preference} 为 {mapped_value}\n\n{get_profile_summary()}"
    except Exception as e:
        return f"更新失败: {e}"
