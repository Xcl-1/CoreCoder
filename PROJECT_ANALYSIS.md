# CoreCoder 项目架构分析

> **CoreCoder v0.4.0** — "the nanoGPT of coding agents"
>
> 极简 AI 编程助手，约 1,081 行引擎代码，14 个 Python 源文件。MIT 协议，已发布 PyPI。
> 作者：Yufeng He（前 Moonshot AI / Kimi）
>
> **设计目标**：一个下午通读全部源码，理解 AI 编程助手的完整工作原理。

---

## 目录

1. [项目概述](#1-项目概述)
2. [核心架构](#2-核心架构)
3. [模块详解](#3-模块详解)
4. [工具系统](#4-工具系统)
5. [类层次结构](#5-类层次结构)
6. [上下文压缩策略](#6-上下文压缩策略)
7. [安全机制](#7-安全机制)
8. [数据流与交互图](#8-数据流与交互图)
9. [设计哲学：刻意留白](#9-设计哲学刻意留白)
10. [测试体系](#10-测试体系)
11. [文档体系](#11-文档体系)
12. [附录：快速参考](#附录快速参考)

---

## 1. 项目概述

### 1.1 是什么

CoreCoder 是一个**教育用途**的 AI 编程助手。它用最少的代码完整实现了一个可用的 AI coding agent 核心引擎，就像 [nanoGPT](https://github.com/karpathy/nanoGPT) 之于大语言模型训练——你不需要读几万行代码才能理解 AI 编程助手是怎么工作的。

### 1.2 对比定位

| 维度 | CoreCoder | Claude Code | aider |
|------|-----------|-------------|-------|
| 引擎代码量 | ~1,081 行 | ~50,000+ 行 | ~20,000+ 行 |
| 设计目标 | 教育 / Fork 基础 | 生产级 | 生产级 |
| 一个下午可通读 | ✅ | ❌ | ❌ |
| 可扩展性 | Fork 即扩展 | 插件/Hook 系统 | 配置驱动 |
| Python 版本 | 3.10+ | N/A | 3.10+ |

### 1.3 技术栈

- **语言**: Python 3.10+
- **构建工具**: hatchling
- **核心依赖**: `openai>=1.0`, `rich>=13.0`, `prompt_toolkit>=3.0`, `python-dotenv>=1.0`
- **可选依赖**: `litellm>=1.60.0`（支持 100+ 非 OpenAI API 提供商）
- **开发依赖**: `pytest>=7.0`, `ruff>=0.9.0`

### 1.4 项目结构

```
corecoder/                      # 主包（引擎核心）
├── __init__.py                 # 导出 Agent, LLM, Config, ALL_TOOLS
├── agent.py                    # 核心 Agent 循环 (151 行) ⭐
├── llm.py                      # LLM 客户端层 (337 行)
├── context.py                  # 三级上下文压缩 (210 行)
├── cli.py                      # REPL / CLI 界面 (270 行)
├── config.py                   # 环境变量配置 (57 行)
├── prompt.py                   # 系统提示词生成 (33 行)
├── session.py                  # 会话持久化 (97 行)
└── tools/                      # 工具实现（7 个工具，~521 行）
    ├── __init__.py             # 工具注册表 (ALL_TOOLS)
    ├── base.py                 # Tool 抽象基类 (27 行)
    ├── bash.py                 # Shell 执行 + 安全门 (127 行)
    ├── edit.py                 # 唯一匹配搜索替换 + diff (92 行) ⭐
    ├── grep.py                 # 正则内容搜索 (79 行)
    ├── glob_tool.py            # 文件名通配 (47 行)
    ├── read.py                 # 文件读取 + 行号 (53 行)
    ├── write.py                # 文件创建/覆盖 (38 行)
    └── agent.py                # 子 Agent 生成 (58 行)

tests/                          # 测试套件 (86 个测试)
├── test_core.py                # 核心 Agent / LLM / 上下文压缩测试
├── test_tools.py               # 全部 7 个工具的功能和并发测试
├── test_session.py             # 会话持久化 + 路径遍历防护测试
└── test_litellm.py             # LiteLLM 后端 + 多提供商测试

article/                        # 8 篇双语源码导读系列
├── 00-index.md / _EN.md        # 导航索引
├── 01-the-loop.md / _EN.md     # Agent 循环
├── 02-tools.md / _EN.md        # 工具系统
├── 03-llm-and-cost.md / _EN.md # LLM 与成本
├── 04-context.md / _EN.md      # 上下文压缩
├── 05-parallel-and-subagents.md / _EN.md  # 并行与子 Agent
├── 06-session-and-cli.md / _EN.md        # 会话与 CLI
└── 07-build-your-own.md / _EN.md         # Fork 实战指南
```

---

## 2. 核心架构

### 2.1 一句话概括

CoreCoder 的整个引擎就是一个 **`while` 循环**：不断把对话历史 + 工具列表发给 LLM，LLM 返回工具调用就执行，返回文本就结束。

### 2.2 核心循环

```
用户输入 → [系统提示词 + 对话历史 + 工具 Schema] → LLM
    → LLM 返回文本? → 追加到历史 → 返回给用户 (结束)
    → LLM 返回 tool_calls? → 追加到历史 → 执行工具 (多个则并行)
        → 工具结果追加到对话历史
        → 必要时压缩上下文
        → 循环
```

这个循环在 [agent.py](corecoder/agent.py) 的 `Agent.chat()` 方法中实现，核心代码不到 50 行：

- 使用 `for _ in range(max_rounds)`（默认 50 轮）而非 `while True`——硬上限防止无限消耗 Token
- 多工具调用通过 `ThreadPoolExecutor(max_workers=8)` 并行执行——模拟 Claude Code 的 StreamingToolExecutor
- Ctrl+C 中断时自动回填 `[interrupted]` 占位回复，保护对话历史不损坏

### 2.3 关键设计决策

| 决策 | 实现 | 为什么 |
|------|------|--------|
| 有上限的循环 | `for _ in range(50)` 而非 `while True` | 防止无限 Token 消耗 |
| 实例级工具注册 | 每个 Agent 有独立的 `_tool_by_name` 字典 | 子 Agent 工具隔离（如去掉 `agent` 防止递归） |
| 两阶段工具执行 | `inspect.signature().bind()` → `execute()` | 区分"参数格式错误"和"工具内部异常"，帮助 LLM 自纠正 |
| 字符估算 Token | `len(text) // 3` | 压缩决策只需近似比例，省去 tokenizer 依赖 |
| 同步子 Agent | ThreadPoolExecutor 而非 async | IO 密集型场景线程够用，简单可靠 |
| Ctrl+C 保护 | 中断后回填占位 tool 回复 | OpenAI API 要求 assistant tool_calls 消息必须有配对 tool 回复 |

---

## 3. 模块详解

### 3.1 Agent 引擎 — [agent.py](corecoder/agent.py) (151 行)

**职责**: 核心 Agent 循环，是整个系统的心脏。将 LLM、工具列表、对话历史、上下文管理器绑定在一起。

```python
class Agent:
    def __init__(self, llm, tools=None, max_context_tokens=128000, max_rounds=50):
        self.llm = llm                          # LLM 客户端
        self.tools = tools or ALL_TOOLS          # 工具列表
        self._tool_by_name = {t.name: t for t in self.tools}  # 实例级工具注册表 ⭐
        self.messages = []                       # 对话历史
        self.context = ContextManager(max_tokens) # 上下文管理器
        self._system = system_prompt(self.tools)  # 系统提示词
```

**核心方法**:

| 方法 | 功能 |
|------|------|
| `chat(user_input, on_token, on_tool) → str` | 处理一条用户消息，内部可能多轮 LLM/工具交互 |
| `_exec_tool(tc) → str` | 执行单个工具调用（先验证参数，再执行） |
| `_exec_tools_parallel(tool_calls) → list[str]` | 线程池并行执行多个工具 (max 8 workers) |
| `_answer_pending_tool_calls(tool_calls)` | 中断后回填占位回复，保持消息历史有效 |
| `reset()` | 清空对话历史 |

**并行工具执行**：当 LLM 在一次响应中返回多个 `tool_calls`（如同时读文件和搜索内容），它们通过 `ThreadPoolExecutor` 并发执行。这模拟了 Claude Code 在模型仍在生成时就开始执行工具的能力。

**中断保护**：如果用户在执行工具时按下 Ctrl+C，`_answer_pending_tool_calls` 确保每个 `tool_call` 都有对应的 `tool` 回复消息。没有这个保护，下一次 LLM 请求会因"孤儿 tool_calls"被 API 拒绝。

---

### 3.2 LLM 层 — [llm.py](corecoder/llm.py) (337 行)

**职责**: 与 LLM 提供商通信，处理流式响应、重试、Token 计数和成本估算。这是最大的文件，因为它需要处理流式 API 的所有现实复杂度。

**关键类**:

| 类 | 说明 |
|-----|------|
| `ToolCall` | 数据类：`id`, `name`, `arguments` |
| `LLMResponse` | 数据类：`content`, `tool_calls`, token 计数；提供 `.message` 属性转为 OpenAI 格式 |
| `LLM` | OpenAI 兼容流式客户端 |
| `LiteLLM(LLM)` | 多提供商后端（100+ API） |

**流式工具调用拼接**：在流式模式下，每个 `tool_call` 的 JSON 参数是**分片到达**的。代码通过 `tc_map` 字典按索引累积这些片段，在所有片段到齐后解析完整 JSON。这是实现流式工具调用的核心复杂度所在。

**重试机制** (`_call_with_retry`)：
- 重试条件：`RateLimitError`, `APITimeoutError`, `APIConnectionError`, 5xx 服务端错误
- 不重试：4xx 客户端错误（请求本身就是错的）
- 退避策略：`2^attempt` 秒，最多 3 次

**成本估算**：内置 `_PRICING` 字典覆盖 20+ 模型（GPT-5.5 系列、DeepSeek、Claude、Qwen、Kimi），`/tokens` 命令展示 Token 用量和估算美元成本。

**LiteLLM 子类**：不创建 OpenAI 客户端，改用 `litellm.completion()` 直接调用。支持 `openai/gpt-4o`、`anthropic/claude-3-haiku`、`bedrock/...`、`ollama/...` 等格式。激活方式：`CORECODER_PROVIDER=litellm`。

---

### 3.3 上下文压缩 — [context.py](corecoder/context.py) (210 行)

**职责**: 在对话历史超出 Token 限制时分层层压缩，防止 API 调用失败。详见[第 6 节](#6-上下文压缩策略)。

---

### 3.4 CLI / REPL — [cli.py](corecoder/cli.py) (270 行)

**职责**: 用户交互界面，使用 Rich + prompt_toolkit 构建。

**两种模式**:

| 模式 | 命令 | 行为 |
|------|------|------|
| 交互 REPL | `corecoder` | 持续对话，支持 8 个斜杠命令 |
| One-shot | `corecoder -p "prompt"` | 执行单次任务后退出，流式输出 |

**斜杠命令**:

| 命令 | 功能 |
|------|------|
| `/help` | 显示帮助 |
| `/reset` | 清空对话历史 |
| `/model [name]` | 查看/切换模型 |
| `/tokens` | Token 用量 + 成本估算 |
| `/compact` | 手动触发上下文压缩 |
| `/diff` | 查看本次会话修改的文件 |
| `/save` | 保存会话到 `~/.corecoder/sessions/` |
| `/sessions` | 列出已保存会话 |
| `quit/exit` | 退出 |

**输入快捷键**:
- `Enter`：提交当前输入
- `Esc+Enter`：插入文字换行（方便粘贴代码块）

**核心/界面分离**：`on_token` 和 `on_tool` 两个回调是 Agent 和 CLI 之间的**唯一耦合点**。Agent 通过回调通知界面，界面负责展示。这意味着 CoreCoder 可以被嵌入到 Web 界面、CI 流水线等任何地方。

---

### 3.5 配置 — [config.py](corecoder/config.py) (57 行)

**职责**: 环境变量驱动的配置系统，零依赖。

```python
@dataclass
class Config:
    model: str = "gpt-5.5"
    api_key: str = ""
    base_url: str | None = None
    max_tokens: int = 4096
    temperature: float = 0.0
    max_context_tokens: int = 128000
    provider: str = "openai"
```

**环境变量映射**（优先级从高到低）：

| 字段 | 环境变量 |
|------|---------|
| `api_key` | `CORECODER_API_KEY` → `OPENAI_API_KEY` → `DEEPSEEK_API_KEY` |
| `model` | `CORECODER_MODEL` |
| `base_url` | `OPENAI_BASE_URL` → `CORECODER_BASE_URL` |
| `max_tokens` | `CORECODER_MAX_TOKENS` |
| `temperature` | `CORECODER_TEMPERATURE` |
| `max_context_tokens` | `CORECODER_MAX_CONTEXT` |
| `provider` | `CORECODER_PROVIDER` |

**.env 文件加载**：从 CWD 向上遍历到 HOME 目录，自动加载第一个找到的 `.env` 文件。

---

### 3.6 系统提示词 — [prompt.py](corecoder/prompt.py) (33 行)

**职责**: 整个系统提示词就是一个 33 行的 Python 函数。

```python
def system_prompt(tools: list[Tool]) -> str:
    """生成系统提示词，包含：Agent 身份、工作目录、OS/Python 版本、
    可用工具列表（动态生成）、八条行为规则"""
```

**八条行为规则**：
1. 先读后改（读文件 → 理解 → 编辑）
2. 小改动用 `edit_file`，大改动用 `write_file`
3. 改完要验证（跑测试/命令）
4. 保持简洁
5. 一步一来
6. `edit_file` 的 `old_string` 要足够唯一
7. 尊重现有代码风格
8. 不确定就问

**设计意图**：没有模板引擎、没有提示词目录、没有 Jinja2。"提示词定制"就是直接改这个函数。项目的文章明确邀请用户："prompt.py is all of 33 lines; change one line and you'll watch the agent's temperament shift."

---

### 3.7 会话持久化 — [session.py](corecoder/session.py) (97 行)

**职责**: 会话的 JSON 保存和恢复。

**存储格式**：`~/.corecoder/sessions/{session_id}.json`

```json
{
  "id": "session_20240726_120000_abc12345",
  "model": "gpt-5.5",
  "saved_at": "2024-07-26 12:00:00",
  "messages": [ ... ]
}
```

**两层安全防护**：

1. **输入净化** (`_normalize_session_id`)：反斜杠转正斜杠 → 取最后一段 → 替换非法字符为 `-` → 100 字符上限
2. **路径验证** (`_session_path`)：`.resolve()` 展平所有 `..` 和符号链接 → 显式检查父目录确实是 `~/.corecoder/sessions/` → 不匹配则抛出 `ValueError`

恶意 ID 如 `../../etc/passwd` 在第一层被净化为 `passwd`，在第二层被路径验证拒绝。

---

## 4. 工具系统

### 4.1 工具基类 — [tools/base.py](corecoder/tools/base.py) (27 行)

```python
class Tool(ABC):
    name: str           # 工具名称（LLM 用这个来调用）
    description: str    # 工具描述
    parameters: dict    # JSON Schema 参数定义

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """执行工具，返回结果字符串"""

    def schema(self) -> dict:
        """组装 OpenAI function-calling 格式的 tool schema"""
```

无继承层次。每个工具直接继承 `Tool`。组合优于继承。

### 4.2 工具注册 — [tools/\_\_init\_\_.py](corecoder/tools/__init__.py) (28 行)

```python
ALL_TOOLS = [
    BashTool(), ReadFileTool(), WriteFileTool(),
    EditFileTool(), GlobTool(), GrepTool(), AgentTool(),
]
```

添加新工具只需两步：创建 `Tool` 子类 → 加入 `ALL_TOOLS` 列表。不需要修改任何其他文件。

### 4.3 工具清单

| # | 工具 | 类 | 文件 | 行数 | 核心能力 |
|---|------|-----|------|------|----------|
| 1 | `bash` | `BashTool` | [bash.py](corecoder/tools/bash.py) | 127 | Shell 执行 + 超时 + 危险命令拦截 + 线程本地 CWD |
| 2 | `read_file` | `ReadFileTool` | [read.py](corecoder/tools/read.py) | 53 | 1-based 行号 + 偏移/限量 + UTF-8 |
| 3 | `write_file` | `WriteFileTool` | [write.py](corecoder/tools/write.py) | 38 | 创建/覆盖 + 自动 mkdir |
| 4 | `edit_file` | `EditFileTool` | [edit.py](corecoder/tools/edit.py) | 92 | **核心创新**：唯一匹配搜索替换 |
| 5 | `grep` | `GrepTool` | [grep.py](corecoder/tools/grep.py) | 79 | 正则搜索 + 跳过垃圾目录 |
| 6 | `glob` | `GlobTool` | [glob_tool.py](corecoder/tools/glob_tool.py) | 47 | 文件名通配 + mtime 排序 |
| 7 | `agent` | `AgentTool` | [agent.py](corecoder/tools/agent.py) | 58 | 同步子 Agent 生成（无递归） |

### 4.4 重点工具详解

#### BashTool — Shell 执行与安全

- **危险命令黑名单**: 10 个正则模式，在执行前检查。覆盖 `rm -rf /`、`mkfs`、`dd to /dev`、fork bomb、curl/wget pipe to shell 等
- **标志重排检测**: 使用前瞻断言，`rm -rf` 和 `rm -fr` 和 `rm -r -f` 都能匹配
- **输出截断**: 超过 15,000 字符时保留前 6,000 + 后 3,000
- **超时控制**: 默认 120 秒
- **线程本地 CWD**: 使用 `threading.local()` 追踪 `cd` 切换。两个并行的 bash 调用互相隔离，不会争抢全局状态
- **链式 cd 解析**: `cd a && cd b` 正确解析为 `a/b`（从运行时目录按序解析）

#### EditFileTool — 唯一匹配搜索替换 ⭐

这是 CoreCoder **最具创新性的工具**。它解决了 LLM 编辑文件的核心痛点：**如何在不传输整个文件的情况下精确定位修改位置**。

工作原理：
1. LLM 提供 `old_string`（要替换的内容片段）和 `new_string`（新内容）
2. 工具在文件中搜索 `old_string`：
   - **0 次匹配** → 返回文件开头预览，帮助 LLM 重新定位
   - **1 次匹配** → 执行替换，返回 unified diff
   - **多次匹配** → 返回"请提供更多上下文以保证唯一性"
3. 修改过的文件记录在全局 `_changed_files` set 中，供 `/diff` 命令查询

不依赖行号（行号会随修改偏移），不依赖完整文件内容（大文件传输成本高）。LLM 只要能记住要修改的代码片段本身即可。

#### AgentTool — 子 Agent 生成

- 创建独立 `Agent` 实例，拥有**独立的对话上下文**
- 工具列表：父 Agent 的工具**减去 `agent` 自身**（实例级注册表保证无法绕过）
- 轮次限制：20 轮（父 Agent 默认 50 轮），防止子任务失控
- 输出截断：超过 5,000 字符时截断到 4,500，防止污染父 Agent 上下文
- **核心价值**：上下文隔离——探索性工作（研究代码库、尝试修改）在子 Agent 窗口中完成，只有蒸馏后的结论返回主对话

---

## 5. 类层次结构

```
Tool (ABC)                     # tools/base.py
├── BashTool                   # tools/bash.py
├── ReadFileTool               # tools/read.py
├── WriteFileTool              # tools/write.py
├── EditFileTool               # tools/edit.py
├── GlobTool                   # tools/glob_tool.py
├── GrepTool                   # tools/grep.py
└── AgentTool                  # tools/agent.py

Config (dataclass)             # config.py

LLM                            # llm.py
└── LiteLLM(LLM)               # llm.py

LLMResponse (dataclass)        # llm.py
ToolCall (dataclass)           # llm.py

ContextManager                 # context.py

Agent                          # agent.py
├── 拥有 LLM 实例
├── 拥有 list[Tool] + _tool_by_name 字典
├── 拥有 ContextManager
├── 拥有 messages (对话历史)
└── 通过 AgentTool._parent_agent 被引用
```

**设计特点**：
- 唯一的继承关系是 `LiteLLM(LLM)`，其余全部通过组合
- 无中间件、无 Hook 系统、无事件总线
- Flat is better than nested——Python 之禅在架构中的体现

---

## 6. 上下文压缩策略

CoreCoder 实现了**三级分层压缩**，灵感来自 Claude Code 的四层系统 (HISTORY_SNIP → Microcompact → CONTEXT_COLLAPSE → Autocompact)。

### 6.1 Token 估算

```python
def _approx_tokens(text: str) -> int:
    return len(text) // 3  # 中英混合约 3 字符/token
```

**为什么不用精确 tokenizer？** 压缩决策只需要近似比例判断（"是否超过 50%?"），不需要精确计数。省去引入 `tiktoken` 等依赖。

### 6.2 三级压缩明细

| 层级 | 触发阈值 | 方法 | LLM 成本 | 说明 |
|------|---------|------|---------|------|
| **L1 Tool Snip** | 50% | 纯文本截断 | 免费 | 工具输出 > 1500 字符 → 保留前 3 行 + `...snip...` + 后 3 行 |
| **L2 Summarize** | 70% + 消息 > 10 条 | LLM 摘要 | 1 次 API 调用 | 旧消息 (除最近 8 条) → LLM 生成摘要 |
| **L3 Hard Collapse** | 90% + 消息 > 4 条 | LLM 摘要 | 1 次 API 调用 | 只留最近 4 条 + 完整摘要（标注 `[Hard context reset]`）|

### 6.3 安全分割机制 (`_safe_split`)

```python
def _safe_split(messages, keep_recent):
    split = max(0, len(messages) - keep_recent)
    while split > 0 and messages[split].get("role") == "tool":
        split -= 1  # 回退！不能从 tool 消息处切割
    return split
```

这是压缩系统中最关键的**安全机制**。OpenAI 兼容 API 要求每条 assistant `tool_calls` 消息之后必须紧跟对应数量的 `tool` 回复。如果在 tool 消息处切割，会产生"孤儿 tool 消息"（没有前置 tool_calls 的 tool 消息），API 会直接拒绝请求。

`_safe_split` 确保切割点永远不会切断 assistant-tool 消息对。

### 6.4 摘要生成

```python
def _get_summary(messages, llm):
    if llm:
        # LLM 生成结构化摘要
        # 保留：文件路径、关键决策、错误信息、任务状态
        # 丢弃：冗长命令输出、代码清单、重复对话
    else:
        # 回退：正则提取文件路径 + 错误行
```

**回退策略**：如果 LLM 调用失败（网络错误等），自动切换到正则提取模式，至少保留文件路径和错误信息。

---

## 7. 安全机制

### 7.1 Bash 危险命令黑名单

10 个正则模式，**在执行前**检查命令字符串：

| 模式示例 | 防护目标 |
|----------|---------|
| `rm -r[f] / ~ $HOME` | 递归删除根/家目录 |
| `rm` + `-r/R` + `-f` (任意标志顺序) | 强制递归删除 |
| `mkfs` | 格式化文件系统 |
| `dd ... of=/dev/` | 原始磁盘写入 |
| `> /dev/sd*` | 覆写块设备 |
| `chmod 777 /` | 根目录权限开放 |
| `:(){ :\|:& };:` | Fork 炸弹 |
| `curl ... \| bash` / `wget ... \| sh` | 下载即执行 |

**定位说明**：项目文档明确指出这是"防止手滑的护栏，不是安全边界"。不可信场景应使用 seccomp 或容器隔离。

### 7.2 会话路径遍历防护

两层纵深防御（详见 [3.7 节](#37-会话持久化--sessionpy-97-行)）。

### 7.3 实例级工具隔离

每个 `Agent` 实例独立维护 `_tool_by_name` 字典。子 Agent 的 `agent` 工具被移除后，即使 LLM 在提示词中看到 `agent` 工具名，调用时也会得到 `"Error: unknown tool 'agent'"`。这是**物理级**限制而非提示词级限制，无法被 LLM 绕过。

---

## 8. 数据流与交互图

```
cli.py / main()
│
├── Config.from_env()                      # config.py — 读取环境变量
├── LLM() 或 LiteLLM()                     # llm.py — 创建 LLM 客户端
│
└── Agent(llm, tools)                      # agent.py — 创建 Agent
    │
    ├── system_prompt(tools)               # prompt.py — 生成系统提示词
    ├── ContextManager(max_tokens)         # context.py — 上下文管理器
    └── ALL_TOOLS                          # tools/__init__.py — 工具注册表
        ├── BashTool.execute()
        ├── ReadFileTool.execute()
        ├── WriteFileTool.execute()
        ├── EditFileTool.execute()         # 维护 _changed_files 全局集合
        ├── GlobTool.execute()
        ├── GrepTool.execute()
        └── AgentTool.execute()
            └── Agent(llm, tools-without-agent)  # 递归创建子 Agent
                └── ... (同上，但没有 agent 工具)

Agent.chat(user_input)
│
├── messages.append({"role": "user", ...})     # 追加用户消息
├── context.maybe_compress(messages, llm)      # 压缩检查入口
│
└── for round in range(max_rounds):            # 核心循环
    ├── llm.chat(messages, tools, on_token)    # LLM 流式调用
    │   ├── OpenAI SDK stream=True
    │   ├── 流式接收 → 拼接工具调用参数片段
    │   ├── 累计 prompt/completion token 计数
    │   └── 返回 LLMResponse
    │
    ├── 无 tool_calls? → append + return text  # 对话结束
    │
    └── 有 tool_calls? →
        ├── 单个: _exec_tool(tc)
        └── 多个: _exec_tools_parallel(tcs)    # ThreadPoolExecutor(max=8)
            └── _exec_tool(tc)
                ├── _tool_by_name.get(name)    # 实例级查找（子Agent隔离的关键）
                ├── signature.bind(**args)     # 参数验证 → 区分参数错和执行错
                └── tool.execute(**args)       # 实际执行
        ├── 追加 tool 结果到 messages           # role: "tool"
        └── context.maybe_compress()           # 执行后再次检查（工具输出可能很大）
```

---

## 9. 设计哲学：刻意留白

CoreCoder **刻意不实现**以下特性。每个空白都是一个明确的扩展入口，是项目教育价值的一部分：

| 缺失特性 | 当前替代方案 | 扩展方向 |
|---------|-------------|---------|
| **安全沙箱** | Bash 正则黑名单 | seccomp / Docker 容器 / 虚拟机隔离 |
| **Skill 系统** | 无，全在系统提示词 | 意图路由 + 角色提示词注入 + 插件加载 |
| **跨会话记忆** | 会话 JSON dump | 记忆提取 → 索引 → 检索 → 注入闭环 |
| **MCP 协议** | 无 | 外部工具协议集成 (Model Context Protocol) |
| **异步子 Agent** | 同步 ThreadPoolExecutor | 后台 Agent + 回调/通知 |
| **权限确认提示** | 所有工具直接执行 | 多层审查链 + cli 交互式提示 |
| **回退模型** | 仅指数退避重试 | 多模型链 + 硬性预算上限 |
| **检查点/撤销** | edit diff 可人工审查 | 操作历史 + 回滚到任意状态 |
| **RAG 检索** | 无 | 向量嵌入 + 代码库语义搜索 |

**项目设计原则**：
1. **可读性 > 完备性**：代码要一个下午能读完
2. **Fork > 配置**：扩展通过 fork 代码而不是改配置
3. **显式 > 隐式**：每种设计决策在文章中有解释
4. **够用就好**：字符估算 Token 而非引入 tokenizer、正则黑名单而非容器

---

## 10. 测试体系

### 10.1 概况

- **框架**: pytest
- **测试文件**: 4 个
- **测试数量**: 86 个，全部通过 ✅
- **CI**: GitHub Actions, 3 OS (Ubuntu/Mac/Windows) × 4 Python (3.10~3.13) 矩阵

### 10.2 测试模式

| 模式 | 示例 | 用途 |
|------|------|------|
| **pytest fixtures** | `tmp_path`, `monkeypatch` | 临时文件、环境变量注入 |
| **直接函数测试** | 导入工具，调用 `execute()`，断言返回值 | 工具功能验证 |
| **Mock 替换** | 创建假 `litellm` 模块注入 `sys.modules` | 隔离外部依赖 |
| **线程安全测试** | 两个线程并发验证 `threading.local` 隔离 | 并行执行正确性 |
| **不变量保护** | 验证压缩后无孤儿 tool 消息 | API 约束保证 |
| **参数化测试** | `pytest.mark.parametrize` 多场景覆盖 | 边界条件覆盖 |

### 10.3 CI 管道

```yaml
strategy:
  matrix:
    os: [ubuntu-latest, macos-latest, windows-latest]
    python-version: ["3.10", "3.11", "3.12", "3.13"]

steps:
  - pytest -v tests/
  - ruff check corecoder/
  - python -m compileall corecoder/
  - python -m build && twine check dist/*
```

---

## 11. 文档体系

CoreCoder 的文档独树一帜——**8 篇中英双语源码导读文章就是它的架构文档**。没有单独的 `/docs/` 目录。

| # | 文章 | 涵盖的源码文件 | 核心主题 |
|---|------|---------------|---------|
| 1 | The Agent Loop | [agent.py](corecoder/agent.py) | 为什么是 for 循环而非 while True |
| 2 | The Tool System | [tools/](corecoder/tools/) | edit_file 唯一匹配的创新设计 |
| 3 | LLM and Cost | [llm.py](corecoder/llm.py) | 流式工具调用拼接的细节 |
| 4 | Context Compression | [context.py](corecoder/context.py) | 孤儿 tool 消息陷阱与 _safe_split |
| 5 | Parallelism & Sub-agents | [agent.py](corecoder/agent.py), [tools/agent.py](corecoder/tools/agent.py) | threading.local() 与线程安全 |
| 6 | Session and CLI | [cli.py](corecoder/cli.py), [session.py](corecoder/session.py) | 路径遍历防护的两层设计 |
| 7 | Build Your Own | 全部文件 | Fork 实战：定制工具、提示词、作为库使用 |
| 0 | Index | — | 阅读顺序建议和导航 |

每篇文章的结构：
1. "Why this matters"——为什么这个模块值得深入学习
2. 源码逐行解读——实际代码片段 + 设计原理解释
3. 边界情况分析——每个设计决策的 trade-off
4. "What this piece leaves you with"——关键收获总结

---

## 附录：快速参考

### 文件行数统计

| 文件 | 行数 | 分类 | 一句话说明 |
|------|------|------|-----------|
| `llm.py` | 337 | LLM 层 | 流式客户端 + 重试 + 成本 + LiteLLM |
| `cli.py` | 270 | CLI | Rich REPL + 8 条斜杠命令 |
| `context.py` | 210 | 上下文 | 三级压缩 + _safe_split |
| `agent.py` | 151 | 核心 | for 循环连接 LLM 与工具 |
| `tools/bash.py` | 127 | 工具 | Shell 执行 + 安全黑名单 |
| `session.py` | 97 | 会话 | JSON 持久化 + 路径防护 |
| `tools/edit.py` | 92 | 工具 | 唯一匹配搜索替换 |
| `tools/grep.py` | 79 | 工具 | 正则内容搜索 |
| `tools/agent.py` | 58 | 工具 | 子 Agent 生成 |
| `config.py` | 57 | 配置 | 环境变量 + .env |
| `tools/read.py` | 53 | 工具 | 文件读取 |
| `tools/glob_tool.py` | 47 | 工具 | 文件名通配 |
| `tools/write.py` | 38 | 工具 | 文件写入 |
| `prompt.py` | 33 | 提示词 | 单函数生成 |
| `tools/__init__.py` | 28 | 注册 | 工具注册表 |
| `tools/base.py` | 27 | 基类 | Tool ABC |
| **引擎总计** | **~1,704** | | 14 个源文件 |

### 快速导航

- **想理解整个 Agent 循环？** → [agent.py](corecoder/agent.py) `Agent.chat()` 方法（核心逻辑 ~55 行）
- **想添加新工具？** → [tools/base.py](corecoder/tools/base.py) 定义接口 + [tools/\_\_init\_\_.py](corecoder/tools/__init__.py) 注册
- **想修改提示词？** → [prompt.py](corecoder/prompt.py) `system_prompt()` 函数（33 行）
- **想换 LLM 提供商？** → [llm.py](corecoder/llm.py) `LLM` 类或 `LiteLLM` 子类
- **想调整压缩策略？** → [context.py](corecoder/context.py) `ContextManager` 的阈值参数
- **想扩展 CLI 命令？** → [cli.py](corecoder/cli.py) `_repl()` 中的 slash command 处理
- **想看完整设计原理？** → [article/](article/) 8 篇双语文章
