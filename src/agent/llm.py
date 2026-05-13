"""LLM 工厂函数"""

from langchain_openai import ChatOpenAI
from config.settings import settings


def create_llm() -> ChatOpenAI:
    """根据配置创建 LLM 实例

    支持两种模式:
    1. OpenAI 兼容接口 (通过 LLM_BASE_URL + LLM_API_KEY)
    2. 标准 OpenAI (通过 OPENAI_API_KEY)

    Returns:
        ChatOpenAI 实例
    """
    if settings.use_openai_compat:
        return ChatOpenAI(
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            temperature=0.3,
            streaming=True,
        )
    else:
        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0.3,
            streaming=True,
        )
