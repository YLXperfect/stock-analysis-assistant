# Stock Analysis Assistant

**5-Agent AI 股票分析系统** — 覆盖查询→分析→策略→选股全流程，流式输出，多轮对话。

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/langchain-1.2-green.svg)](https://www.langchain.com/)
[![LangGraph](https://img.shields.io/badge/langgraph-1.1-orange.svg)](https://www.langchain.com/langgraph)

## 快速开始

```bash
git clone <repo>
cd stock-analysis-assistant

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 配置
cp .env.example .env
# 编辑 .env 填入 LLM_API_KEY

# 启动交互模式
python -m src.main -i

# 单次查询
python -m src.main -q "分析特斯拉"
```

## 架构

```
用户输入 → Router（纯意图识别，无工具）
          ├→ Query Agent ──── 数据查询和展示 ─────────────→ END
          ├→ Analyst ────→ Strategist ── 分析+策略建议 ──→ END
          ├→ Strategist（直接）── 交易策略/持仓管理 ──────→ END
          └→ Screener ───────────── 条件选股 ───────────→ END
```

5 个 Agent 各有独立 prompt 和工具集，通过 LangGraph `StateGraph` 条件路由编排。

| Agent | 职责 | 工具 |
|-------|------|------|
| **Router** | 意图识别（关键词 + LLM 兜底），越界拒绝 | 无 |
| **Query** | 行情/资金/新闻/持仓/自选股等数据查询和展示 | 30+ |
| **Analyst** | 技术面/基本面/资金面/新闻多维分析 | 18 |
| **Strategist** | 多空辩论、交易提案、持仓安全、策略回测 | 12 |
| **Screener** | 条件选股（PE/PB/ROE/市值等） | 5 |

## 功能

| 类别 | 功能 |
|------|------|
| 行情查询 | 实时报价、K线(日/周/月/分钟)、盘口深度、分时数据 |
| 资金面 | 主力/散户资金流向 |
| 基本面 | ROE/利润率/负债/现金流/估值 6 维加权评分 |
| 技术面 | MA/MACD/RSI/布林带/ATR，1-10 综合评分 |
| 策略 | 多空辩论、入场/止损/止盈/仓位建议 |
| 风控 | 持仓安全评估、组合风险扫描 |
| 回测 | 均线交叉/RSI/布林带/MACD 4 种策略模板 |
| 选股 | PE/PB/市值/ROE/营收增长条件筛选（美股 52 只 + 港股 34 只） |
| 新闻 | 个股新闻、机构评级、内幕交易（美股） |
| 账户 | 持仓查询、资产概览、自选股管理、价格提醒 |
| 记忆 | 多轮对话上下文、用户偏好持久化、交易决策日志 |

## 使用示例

```
You> 查看特斯拉的资金流向          → Query Agent 直接展示数据
You> 分析特斯拉                    → Analyst 分析 → Strategist 输出策略
You> 特斯拉该不该买                → Strategist 多空辩论 + 交易提案
You> 筛选 PE 小于 20 的美股        → Screener 批量筛选
You> 查看特斯拉股价，分析一下行情    → Analyst 先展示数据再分析
```

## 技术栈

| 层级 | 技术 |
|------|------|
| 编排 | LangGraph StateGraph + InMemorySaver |
| Agent | LangChain `create_agent` + `@tool` 装饰器 |
| LLM | ChatOpenAI 兼容接口（GLM-4-Flash / GPT-4o-mini），流式输出 |
| 数据源 | Longbridge Python SDK v4 + yfinance（K线数据） |
| 分析引擎 | pandas + numpy 纯函数（确定性评分，不依赖 LLM） |
| 持久化 | Pydantic JSON + JSONL 追加日志 |
| UI | Rich 终端（Live 流式渲染 + Markdown 面板） |

## 项目结构

```
src/
├── agent/
│   ├── multi_agent.py    # 5-Agent StateGraph 编排
│   └── llm.py            # LLM 工厂（支持 OpenAI 兼容接口）
├── analysis/
│   ├── technical.py      # 技术指标计算（MA/MACD/RSI/布林带/ATR）
│   ├── fundamental.py    # 基本面评分引擎（6 维加权）
│   ├── strategy.py       # 多空辩论 + 交易提案 + 定投策略
│   ├── position.py       # 持仓安全评估
│   └── backtest.py       # PineScript V6 回测模板
├── tools/
│   ├── longbridge_sdk.py # 长桥 SDK 工具封装（行情/基本面/新闻/账户）
│   ├── analysis_tools.py # 分析编排工具（技术面/基本面/策略/回测）
│   └── profile_tools.py  # 用户偏好管理
├── memory/
│   ├── profile.py        # 用户偏好（JSON）
│   └── journal.py        # 决策日志（JSONL）
├── skills/               # 分析方法论 Skill
└── main.py               # CLI 入口（流式输出 + 交互模式）
config/
└── settings.py           # 配置管理（Pydantic Settings）
data/
├── stock_universe.json   # 股票池（美股 52 + 港股 34）
├── user_profile.json     # 用户偏好
└── journal.jsonl         # 决策日志
```

## 上下文管理

三层记忆机制：

```
短记忆:   InMemorySaver checkpointer → 同一会话跨轮 messages 自动累积
压缩记忆: token > 50K 时 LLM 自动摘要旧消息
长期记忆: data/user_profile.json + data/journal.jsonl → 跨会话持久化
```

## 配置

编辑 `.env` 文件：

```bash
# LLM（二选一）
LLM_API_KEY=your_key          # OpenAI 兼容接口（如智谱 GLM）
LLM_MODEL=glm-4-flash
LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4

# 或使用标准 OpenAI
OPENAI_API_KEY=your_key
OPENAI_MODEL=gpt-4o-mini

# 长桥 SDK（自动复用 CLI 的 OAuth token，无需额外配置）
```

## License

MIT
