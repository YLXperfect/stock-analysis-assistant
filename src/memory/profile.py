"""用户风险偏好持久化

存储用户投资风格、风险承受能力、默认止损止盈等偏好。
文件路径: data/user_profile.json
"""

import json
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field

DATA_DIR = Path(__file__).parent.parent.parent / "data"
PROFILE_FILE = DATA_DIR / "user_profile.json"


class UserProfile(BaseModel):
    risk_tolerance: str = "moderate"        # conservative / moderate / aggressive
    investment_style: str = "growth"        # growth / value / dividend / balanced
    preferred_markets: list[str] = Field(default_factory=lambda: ["HK", "US"])
    max_position_pct: float = 20.0          # 单只最大仓位占比 %
    stop_loss_default_pct: float = 8.0      # 默认止损 %
    take_profit_default_pct: float = 20.0   # 默认止盈 %
    preferred_sectors: list[str] = Field(default_factory=list)
    avoid_sectors: list[str] = Field(default_factory=list)
    notes: str = ""


def ensure_data_dir():
    """确保 data 目录存在"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_profile() -> UserProfile:
    """加载用户偏好，不存在则创建默认"""
    ensure_data_dir()
    if PROFILE_FILE.exists():
        try:
            data = json.loads(PROFILE_FILE.read_text(encoding="utf-8"))
            return UserProfile(**data)
        except (json.JSONDecodeError, Exception):
            pass
    # 返回默认并保存
    profile = UserProfile()
    save_profile(profile)
    return profile


def save_profile(profile: UserProfile):
    """保存用户偏好"""
    ensure_data_dir()
    PROFILE_FILE.write_text(
        profile.model_dump_json(indent=2),
        encoding="utf-8",
    )


def update_profile(**kwargs) -> UserProfile:
    """部分更新用户偏好"""
    profile = load_profile()
    for key, value in kwargs.items():
        if hasattr(profile, key):
            # 处理逗号分隔的列表值
            if key in ("preferred_sectors", "avoid_sectors") and isinstance(value, str):
                value = [s.strip() for s in value.split(",") if s.strip()]
            elif key in ("max_position_pct", "stop_loss_default_pct", "take_profit_default_pct") and isinstance(value, str):
                try:
                    value = float(value)
                except ValueError:
                    continue
            setattr(profile, key, value)
    save_profile(profile)
    return profile


def get_profile_summary() -> str:
    """返回中文摘要，用于注入 Agent prompt"""
    profile = load_profile()
    style_map = {"growth": "成长型", "value": "价值型", "dividend": "红利型", "balanced": "均衡型"}
    risk_map = {"conservative": "保守", "moderate": "稳健", "aggressive": "激进"}

    lines = [
        f"- 风险承受: {risk_map.get(profile.risk_tolerance, profile.risk_tolerance)}",
        f"- 投资风格: {style_map.get(profile.investment_style, profile.investment_style)}",
        f"- 关注市场: {', '.join(profile.preferred_markets)}",
        f"- 单只最大仓位: {profile.max_position_pct}%",
        f"- 默认止损: {profile.stop_loss_default_pct}%",
        f"- 默认止盈: {profile.take_profit_default_pct}%",
    ]
    if profile.preferred_sectors:
        lines.append(f"- 偏好行业: {', '.join(profile.preferred_sectors)}")
    if profile.avoid_sectors:
        lines.append(f"- 回避行业: {', '.join(profile.avoid_sectors)}")
    if profile.notes:
        lines.append(f"- 备注: {profile.notes}")
    return "\n".join(lines)
