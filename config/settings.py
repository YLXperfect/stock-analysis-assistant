"""项目配置管理

数据源: longbridge MCP (Model Context Protocol)
LLM: OpenAI 兼容接口
"""

from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    # ── LLM ────────────────────────────────────
    llm_base_url: str = Field(
        default="https://open.bigmodel.cn/api/paas/v4",
        alias="LLM_BASE_URL",
    )
    llm_api_key: str = Field(default="", alias="LLM_API_KEY")
    llm_model: str = Field(default="glm-4-flash", alias="LLM_MODEL")

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")

    # ── 长桥 MCP ─────────────────────────────────
    longbridge_mcp_url: str = Field(
        default="https://openapi.longbridge.com/mcp",
        alias="LONGBRIDGE_MCP_URL",
    )
    trading_enabled: bool = Field(
        default=False,
        alias="TRADING_ENABLED",
    )

    # ── 数据目录 ────────────────────────────────────
    data_dir: str = Field(
        default="data",
        alias="DATA_DIR",
    )

    @property
    def use_openai_compat(self) -> bool:
        """是否使用 OpenAI 兼容接口"""
        return bool(self.llm_api_key)

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
