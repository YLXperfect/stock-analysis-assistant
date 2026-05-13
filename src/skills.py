"""Skill 文件加载器

从 skills/ 目录加载长桥 Skill 指令文件，注入到 Agent 的 system prompt。

Skill 是一种"指令即知识"的 AI 集成方式：
- 外部 Markdown 文件描述了工具的能力和用法
- Agent 运行时加载这些指令，理解如何使用工具
- 无需硬编码，更新 Skill 文件即可升级 Agent 知识

用法:
    from src.skills import load_skill

    skill_content = load_skill("longbridge")
    # skill_content 是完整的 Skill 指令文本，可注入到 system prompt
"""

from pathlib import Path

SKILLS_DIR = Path(__file__).parent.parent / "skills"


def load_skill(skill_name: str) -> str:
    """加载 Skill 主文件 (SKILL.md)

    Args:
        skill_name: Skill 名称，如 "longbridge"

    Returns:
        Skill 指令的 Markdown 文本
    """
    skill_file = SKILLS_DIR / skill_name / "SKILL.md"
    if not skill_file.exists():
        raise FileNotFoundError(f"Skill 文件不存在: {skill_file}")
    return skill_file.read_text(encoding="utf-8")


def load_skill_reference(skill_name: str, ref_path: str) -> str:
    """加载 Skill 引用文件

    Args:
        skill_name: Skill 名称
        ref_path: 引用文件的相对路径，如 "references/cli/overview.md"

    Returns:
        引用文件的 Markdown 文本
    """
    ref_file = SKILLS_DIR / skill_name / ref_path
    if not ref_file.exists():
        raise FileNotFoundError(f"引用文件不存在: {ref_file}")
    return ref_file.read_text(encoding="utf-8")


def list_skills() -> list[str]:
    """列出所有可用的 Skill"""
    if not SKILLS_DIR.exists():
        return []
    return [d.name for d in SKILLS_DIR.iterdir() if d.is_dir() and (d / "SKILL.md").exists()]


def build_skill_prompt(skill_name: str = "longbridge") -> str:
    """构建完整的 Skill 指令文本，用于注入 Agent system prompt

    加载 SKILL.md 主文件 + CLI overview 引用文件（Agent 最需要的部分）。

    Args:
        skill_name: Skill 名称

    Returns:
        合并后的 Skill 指令文本
    """
    parts = []

    # 主 Skill 文件
    try:
        parts.append(load_skill(skill_name))
    except FileNotFoundError:
        return ""

    # CLI overview（最关键的引用 — 告诉 Agent CLI 怎么用）
    try:
        cli_ref = load_skill_reference(skill_name, "references/cli/overview.md")
        parts.append("\n\n---\n\n## CLI Reference\n\n" + cli_ref)
    except FileNotFoundError:
        pass

    return "\n".join(parts)
