# CoreCoder 五大方向优化计划书

> **目标项目**: CoreCoder v0.4.0（1,081 行引擎代码，14 个源文件）
>
> **原则**: 保持极简哲学、向后兼容、渐进式实施、文件级存储、Python 3.10+ 类型注解

---

## 目录

1. [总体原则与路线图](#1-总体原则与路线图)
2. [优化方向一：Skill 能力体系](#2-优化方向一skill-能力体系)
3. [优化方向二：自进化记忆沉淀](#3-优化方向二自进化记忆沉淀)
4. [优化方向三：分层上下文压缩](#4-优化方向三分层上下文压缩)
5. [优化方向四：多 Agent 协作](#5-优化方向四多-agent-协作)
6. [优化方向五：权限与安全审查](#6-优化方向五权限与安全审查)
7. [文件变更汇总](#7-文件变更汇总)
8. [测试策略](#8-测试策略)
9. [验证方案](#9-验证方案)

---

## 1. 总体原则与路线图

### 1.1 设计原则

| 原则 | 说明 |
|------|------|
| **保持极简** | 每个优化模块应独立、可读、可 fork，新增代码量控制在原引擎的 1~1.5 倍 |
| **向后兼容** | 不传新参数的 `Agent()` 行为与 v0.4.0 完全一致 |
| **渐进式实施** | 5 个方向按优先级分三阶段，每阶段可独立交付 |
| **文件级存储** | 沿用项目的文件存储模式（JSON + Markdown），不引入外部数据库 |
| **代码风格一致** | dataclass、类型注解、docstring、单行不超过 130 字符 |

### 1.2 三阶段路线图

```
第一阶段 (MVP 基础框架) — 预计 2-3 周
├── Skill: 基类 + 注册中心 + 关键词路由 + 3 个内置 Skill
├── Memory: 数据模型 + 文件 CRUD + MEMORY.md 索引
└── Security: PermissionManager + 权限配置文件 + Guard 基础版

第二阶段 (深化优化) — 预计 2-3 周
├── Context: 精确 Token 计数 + L1 语义截断 + L2 增量摘要
├── Multi-Agent: AgentRole 预设角色 + 增强 AgentTool + 异步执行
└── Security: AuditLogger + 用户确认交互 + 决策学习

第三阶段 (高级特性) — 预计 2-3 周
├── Skill: 文件自动加载 + 依赖解析 + Skill 组合
├── Memory: 提取器 + 检索器 + 完整闭环
├── Multi-Agent: Orchestrator + 协作模式 + WorktreeManager
├── Context: L2.5 结构化摘要 + 动态阈值
└── Security: SandboxPolicy + ContentReviewer
```

### 1.3 模块依赖关系

```
Phase 1 (独立，可并行建设):
  ├── Skill system       (corecoder/skills/)
  ├── Memory engine      (corecoder/memory/)
  └── Security guard     (corecoder/security/)

Phase 2 (依赖 Phase 1 部分模块):
  ├── Multi-agent collab (corecoder/collab/)  ──引用──→ Security (worker sandbox)
  └── Context enhance    (修改 context.py)    ──独立──→ 无外部依赖

Phase 3 (集成与串联):
  └── 全部模块接入 agent.py + cli.py
```

---

## 2. 优化方向一：Skill 能力体系

> **可行性**: ⭐⭐⭐⭐⭐ 极高 | **当前状态**: 无 Skill 概念，工具是扁平列表
>
> **目标**: 从零构建分层意图路由 + 提示词注入的 Skill 系统

### 2.1 当前状态分析

- 工具的注册、匹配、执行全部在一个平面上（`ALL_TOOLS` 列表）
- 系统提示词是单一静态函数（[prompt.py](corecoder/prompt.py) 33 行）
- 无意图识别、无角色切换、无插件机制
- 用户无法在不修改源码的情况下定制 Agent 行为

### 2.2 架构设计

```
corecoder/skills/
├── __init__.py          # 公开 API: SkillRouter, Skill, SkillMatch, BUILTIN_SKILLS
├── base.py              # Skill 抽象基类 + SkillMatch 数据类
├── registry.py          # SkillRegistry: 注册、查找、匹配、依赖解析
├── router.py            # SkillRouter: 意图路由 + 提示词增补生成
├── loader.py            # SkillLoader: 从 .skill.md 文件加载用户自定义 Skill
└── builtins/
    ├── __init__.py       # BUILTIN_SKILLS 列表
    ├── code_review.py    # 代码审查 Skill
    ├── test_gen.py       # 测试生成 Skill
    └── refactor.py       # 重构 Skill
```

### 2.3 核心设计思想

**Skill 是元工具（Meta-tool）**：Skill 不直接调用工具，它注入**专门的系统提示词片段**告诉 LLM 如何用现有工具完成特定任务。这意味着 Skill 层非常薄——不需要执行引擎，LLM + 现有工具就是执行引擎。

### 2.4 核心数据结构

```python
@dataclass
class SkillMatch:
    """单次匹配结果"""
    skill: Skill
    confidence: float          # 0.0 ~ 1.0
    matched_trigger: str       # 命中的触发模式

class Skill(ABC):
    name: str                  # 唯一标识，如 "code-review"
    description: str           # 一句话描述
    triggers: list[str]        # 正则触发模式，如 [r"\breview\b", r"\baudit\b"]
    keywords: list[str]        # 关键词（权重较低，累积计分）
    prompt_fragment: str       # 激活时注入的提示词片段
    priority: int = 0          # 优先级（越高越优先）
    requires: list[str] = []   # 依赖的其他 Skill 名称
    max_rounds: int = 30       # 该 Skill 的对话轮次上限

    def matches(self, user_input: str) -> SkillMatch | None: ...
    def apply(self, agent) -> None: ...  # 激活时的回调
```

### 2.5 路由算法

```
用户输入 → 遍历所有 Skill → 对每个 Skill:
    1. 正则匹配 triggers: 命中 → confidence = 0.8
    2. 关键词匹配 keywords: 每个命中 +0.15 (上限 0.7)
    3. priority 加成: + priority * 0.05
    → 取最高分 → 低于 min_confidence (默认 0.3) 则丢弃
→ 按 confidence 降序排列 → 返回匹配列表
```

### 2.6 集成方式

**Agent 层改动** ([agent.py](corecoder/agent.py))：

```python
# 构造函数新增参数
def __init__(self, ..., skill_router: SkillRouter | None = None):
    self.skill_router = skill_router

# chat() 方法首轮注入
def chat(self, user_input, ...):
    self.messages.append({"role": "user", "content": user_input})
    # 首条消息时触发 Skill 路由
    if self.skill_router and len(self.messages) <= 1:
        matches = self.skill_router.route(user_input)
        if matches:
            augment = self.skill_router.build_prompt_augment(matches)
            self.messages.insert(-1, {"role": "system", "content": augment})
```

**CLI 层改动** ([cli.py](corecoder/cli.py))：新增 `/skills` 命令列出已注册 Skill 及当前活跃 Skill。

### 2.7 实施阶段

| 阶段 | 内容 | 预估代码量 |
|------|------|-----------|
| Phase 1 | `Skill` 基类 + `SkillMatch` + `SkillRegistry` + 关键词路由 | ~120 行 |
| Phase 2 | 3 个内置 Skill (code-review, test-gen, refactor) | ~120 行 |
| Phase 3 | `SkillLoader`: 从 `~/.corecoder/skills/*.skill.md` 自动加载 | ~80 行 |
| Phase 4 | 依赖解析 + Skill 组合执行 | ~40 行 |

---

## 3. 优化方向二：自进化记忆沉淀

> **可行性**: ⭐⭐⭐⭐⭐ 极高 | **当前状态**: 无跨会话记忆，会话仅是对话历史 JSON dump
>
> **目标**: 设计完整的"提取→存储→检索→注入→更新"记忆沉淀闭环

### 3.1 当前状态分析

- [session.py](corecoder/session.py) 97 行：唯一的持久化是将整个对话序列化为 JSON
- 无记忆提取：会话关闭后，所有上下文丢失
- 无跨会话学习：用户每次都需要重新交代偏好和项目背景
- 无知识积累：Agent 每次从零开始

### 3.2 架构设计

```
corecoder/memory/
├── __init__.py          # MemoryEngine 门面 + 公开 API
├── models.py            # Memory 数据模型 + MemoryType
├── store.py             # MemoryStore: 文件级 CRUD
├── index.py             # MemoryIndex: MEMORY.md 索引维护
├── extractor.py         # MemoryExtractor: LLM 驱动的对话分析提取
├── retriever.py         # MemoryRetriever: 关键词相关性检索
└── engine.py            # MemoryEngine: 串联提取/存储/检索的生命周期门面
```

### 3.3 记忆生命周期闭环

```
┌─────────────────────────────────────────────────────────┐
│                      记忆沉淀闭环                         │
│                                                         │
│  会话结束 ──→ MemoryExtractor ──→ LLM 分析对话           │
│                 │                    │                   │
│                 │              提取关键信息               │
│                 │                    │                   │
│                 ▼                    ▼                   │
│           与已有记忆去重/合并 ──→ MemoryStore 写入文件     │
│                                       │                 │
│                                       ▼                 │
│                              更新 MEMORY.md 索引         │
│                                                         │
│  新会话开始 ──→ MemoryRetriever ──→ 关键词检索            │
│                      │                  │               │
│                      ▼                  ▼               │
│               返回 top-K 记忆 ──→ 注入系统提示词          │
│                                                         │
│  用户纠正 Agent ──→ 生成 feedback 类型记忆               │
│                      ──→ 下次自动遵循                    │
└─────────────────────────────────────────────────────────┘
```

### 3.4 核心数据结构

```python
MemoryType = Literal["user", "feedback", "project", "reference"]

@dataclass
class Memory:
    id: str                    # kebab-case slug, 唯一标识
    title: str                 # 一句话标题（用于展示和检索）
    description: str           # 一行描述（用于 MEMORY.md 索引）
    content: str               # 正文，「Why:」「How to apply:」段
    type: MemoryType           # 记忆类型
    metadata: dict             # {source_session, confidence, ...}
    created_at: str            # ISO 时间戳
    updated_at: str            # ISO 时间戳
    links: list[str]           # [[wiki-link]] 关联的其他记忆
```

### 3.5 存储方案

沿用项目现有的 `~/.corecoder/` 目录体系，每个记忆一个 Markdown 文件：

```markdown
---
name: prefer-pytest-fixtures
description: User prefers pytest fixtures over unittest
metadata:
  type: user
  confidence: 0.9
  source_session: session_20240726_120000_abc12345
  created_at: "2024-07-26T10:00:00Z"
  updated_at: "2024-07-26T10:00:00Z"
---

Always use pytest fixtures (tmp_path, monkeypatch) rather than
unittest.TestCase or manual setup/teardown functions.

**Why:** User explicitly rejected a unittest approach and asked
to rewrite with pytest style.

**How to apply:** When generating tests, default to pytest
fixtures. See also: [[use-pytest-parametrize]]
```

**无外部依赖**：检索使用纯关键词 Token 匹配 + Jaccard 相似度，不引入向量数据库或 embedding 模型。

### 3.6 检索算法

```
输入: 用户的首条消息
1. Tokenize: 分词 → 去停用词 → 小写
2. 对每条记忆计算得分:
   score = (query_tokens ∩ memory_tokens) / |query_tokens|   # 关键词重叠
         + 标题精确匹配 bonus (+0.1 per token)
         + log(1 + recall_count) × 0.05                       # 频率加权
   score *= type_weight[memory.type]                          # 类型加权
3. 按得分排序 → 返回 top-K (默认 5)
```

### 3.7 集成方式

**Agent 层**：构造时注入 `MemoryEngine`，首条消息时检索记忆并注入提示词。

**Session 层**：`save_session()` 后自动触发 `memory.on_session_end()`。

**CLI 层**：
- `/save` 和 `quit` 时触发记忆提取
- 新增 `/memory` 命令：查看记忆统计、最近记忆列表

### 3.8 实施阶段

| 阶段 | 内容 | 预估代码量 |
|------|------|-----------|
| Phase 1 | `Memory` 数据模型 + `MemoryStore` 文件 CRUD + `MemoryIndex` | ~150 行 |
| Phase 2 | `MemoryExtractor` (LLM 提取 + 回退正则) | ~80 行 |
| Phase 3 | `MemoryRetriever` (关键词检索 + 评分) + `MemoryEngine` 门面 | ~120 行 |
| Phase 4 | 去重合并 + 过期衰减 + 置信度管理 | ~60 行 |

---

## 4. 优化方向三：分层上下文压缩

> **可行性**: ⭐⭐⭐⭐ 高 | **当前状态**: 三级压缩存在但实现较基础
>
> **目标**: 在现有三级压缩基础上深化策略，不改变外部接口

### 4.1 当前状态分析

[context.py](corecoder/context.py) 210 行实现了三级压缩：

| 层级 | 触发 | 方法 | 不足 |
|------|------|------|------|
| L1 | 50% | 机械前 3 + 后 3 行截断 | 不区分工具类型，grep 结果和 bash 输出同等对待 |
| L2 | 70% | 一次性 LLM 摘要全部旧消息 | 每次触发都重新摘要全部历史，O(n²) 成本 |
| L3 | 90% | 保留 4 条 + 摘要 | 关键决策点（文件修改、错误）可能被丢弃 |
| 基础设施 | — | `len(text)//3` 估算 Token | 不够精确，可能过早/过晚触发压缩 |

### 4.2 优化方案

所有改动集中在 [context.py](corecoder/context.py)，`ContextManager.maybe_compress()` 的外部接口保持不变。

#### 4.2.1 基础设施：精确 Token 计数

```python
def estimate_tokens_precise(messages: list[dict]) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return sum(len(enc.encode(m.get("content", ""))) for m in messages)
    except ImportError:
        return estimate_tokens(messages)  # 回退到 len//3
```

`tiktoken` 作为**可选依赖**，不安装时自动回退到现有字符估算。

#### 4.2.2 L1 增强：工具类型感知截断

```python
TOOL_SNIP_STRATEGIES = {
    "bash":       "保留前 50 行 + 后 20 行",       # 输出主要是噪声
    "read_file":  "保留前 20 行 + 后 20 行",       # 内容长度重要
    "grep":       "保留全部匹配 (上限 100 条)",     # 每个匹配都可能有用
    "edit_file":  "完整保留",                      # diff 通常很短
    "write_file": "完整保留",                      # 确认信息很短
    "glob":       "完整保留",                      # 结果很少
    "agent":      "保留前 10 行 + 后 10 行",       # 子 Agent 摘要
}
```

回退：未知工具使用现有的前 3 + 后 3 行策略。

#### 4.2.3 L2 增强：增量摘要

现有问题：每次触发 L2 都把**全部**旧消息发给 LLM 重新摘要，成本 O(n)。

改进方案：

```python
class ContextManager:
    _incremental_summary: str = ""        # 累积摘要
    _summary_covers_up_to: int = 0         # 摘要覆盖到的消息索引

    def _summarize_incremental(self, messages, llm, split_idx):
        # 只摘要新增部分 (summary_covers_up_to ~ split_idx)
        new_turns = messages[self._summary_covers_up_to:split_idx]
        # LLM: 将新增内容合并到已有摘要
        merge_prompt = f"Existing summary:\n{self._incremental_summary}\n\nNew:\n{flatten(new_turns)}\n\nUpdate the summary."
        # ...
        self._incremental_summary = new_summary
        self._summary_covers_up_to = split_idx
```

成本从 O(n²) 降为 O(n)。每次摘要只处理增量消息（约 4-8 轮），而不是整个历史。

#### 4.2.4 新增 L2.5：结构化分层摘要

在 L2 和 L3 之间增加一级，对信息分层保留：

| 重要性 | 内容 | 策略 |
|--------|------|------|
| **Critical** | 文件修改记录、错误信息、用户决策 | 完整保留 |
| **Important** | 工具调用参数、返回状态码 | 一行摘要 |
| **Context** | 命令输出、搜索结果 | 丢弃或一行 |

#### 4.2.5 动态阈值

```python
def _compute_thresholds(self, messages):
    rounds = sum(1 for m in messages if m.get("role") == "user")
    if rounds < 5:     return (0.70, 0.85, 0.95)  # 简单任务: 少压缩
    elif rounds < 20:  return (0.50, 0.70, 0.90)  # 当前默认
    else:              return (0.40, 0.60, 0.85)  # 复杂任务: 积极压缩
```

### 4.3 实施阶段

| 阶段 | 内容 | 预估代码量 |
|------|------|-----------|
| Phase 1 | 精确 Token 计数 (tiktoken 可选依赖) | ~30 行 |
| Phase 2 | L1 语义截断 (工具类型感知) + L2 增量摘要 | ~120 行 |
| Phase 3 | L2.5 结构化分层摘要 + 动态阈值 | ~80 行 |

---

## 5. 优化方向四：多 Agent 协作

> **可行性**: ⭐⭐⭐⭐⭐ 极高 | **当前状态**: 仅同步子 Agent，无角色、无编排、无隔离
>
> **目标**: 扩展为中心化多 Agent 协作架构

### 5.1 当前状态分析

[AgentTool](corecoder/tools/agent.py) 58 行是目前唯一的多 Agent 机制：

```python
class AgentTool(Tool):
    def execute(self, task: str) -> str:
        sub = Agent(llm=parent.llm, tools=[t for t in parent.tools if t.name != "agent"],
                    max_rounds=20)
        return sub.chat(task)  # 同步阻塞
```

**局限性**：
- **无角色分工**：所有子 Agent 用同一个系统提示词
- **无并发**：同步执行，父 Agent 阻塞等待
- **无文件隔离**：多个子 Agent 操作同一工作目录会冲突
- **无通信**：Agent 之间无法传递消息或共享上下文
- **单层**：子 Agent 不能再生成子 Agent

### 5.2 架构设计

```
corecoder/collab/
├── __init__.py          # Orchestrator + 公开 API
├── orchestrator.py      # Orchestrator: 中心化任务分解/调度/合成
├── roles.py             # AgentRole: 预设角色模板
├── patterns.py          # 协作模式: Pipeline/Parallel/Debate/Verify
├── worktree.py          # WorktreeManager: Git worktree 文件隔离
└── task.py              # AgentTask / TeamResult 数据类

corecoder/tools/agent.py # 增强: 支持角色 + 异步 + worktree + 工具限定
```

### 5.3 核心数据结构

```python
@dataclass
class AgentRole:
    """预设 Agent 角色模板"""
    name: str                    # "planner" | "executor" | "reviewer" | "researcher"
    system_prompt: str           # 角色专属系统提示词
    tools: list[str] | None      # 可用工具名称列表 (None = 全部)
    max_rounds: int = 20
    can_spawn: bool = False      # 是否可生成子 Agent

@dataclass
class AgentTask:
    """编排器分解出的子任务"""
    id: str
    role: str                    # 需要的角色名称
    description: str             # 任务描述
    context: dict                # 上下文（文件列表、约束条件等）
    dependencies: list[str]      # 依赖的 task ID 列表
    status: Literal["pending", "running", "done", "failed"]

@dataclass
class TeamResult:
    task_id: str
    result: str
    agent_id: str
    tokens_used: int
    duration_ms: int
    files_modified: list[str]
```

### 5.4 协作模式

```
Pipeline (串行管道):      Task1 → Task2 → Task3
    适用: 分步骤的多阶段工作流

Parallel (并行执行):      Task1 ↘
                          Task2 → Merge (合并结果)
                          Task3 ↗
    适用: 独立子任务、多角度搜索

Debate (多方案辩论):      Agent1 ↘
                          Agent2 → Judge → Winner (选最优方案)
                          Agent3 ↗
    适用: 架构决策、有多种实现路径的任务

Verify (执行+审查):       Executor → Reviewer → Approve/Reject
    适用: 高风险操作（如安全相关代码修改）
```

### 5.5 Orchestrator（编排器）

```python
class Orchestrator:
    """中心化多 Agent 编排器"""
    def plan(self, user_input: str) -> list[AgentTask]:
        """LLM 驱动的任务分解，产出 DAG"""

    def execute(self, tasks: list[AgentTask]) -> dict[str, TeamResult]:
        """按 DAG 拓扑序调度（基于依赖的并行）"""

    def synthesize(self, results: dict[str, TeamResult]) -> str:
        """LLM 驱动的结果合成"""
```

### 5.6 Git Worktree 隔离

```python
class WorktreeManager:
    """为每个写文件的 Agent 创建独立 git worktree"""
    def create(self, branch_name: str) -> str:
        """git worktree add .claude/worktrees/{branch} → 返回路径"""
    def merge_back(self, branch_name: str) -> bool:
        """git merge {branch}，处理冲突"""
    def remove(self, branch_name: str, discard: bool = False):
        """清理 worktree"""
```

**回退**：非 Git 项目使用临时目录复制 + diff/patch 方案。

### 5.7 增强 AgentTool

```python
class AgentTool(Tool):
    # 新增参数
    async_mode: bool = False     # 异步? 返回 task_id 而非结果
    role: str | None = None      # 预设角色
    worktree: bool = False       # 是否 worktree 隔离
    tools: list[str] | None = None  # 限定可用工具

    def execute(self, task: str, **kwargs) -> str:
        if self.async_mode:
            return f"[Task submitted: {task_id}]"
        # ...
```

### 5.8 集成方式

**Agent 层**: 构造时可选注入 `Orchestrator`。

**CLI 层**:
- 新增 `/team` 命令触发多 Agent 模式
- 新增 `/tasks` 命令查看运行中的子任务

### 5.9 实施阶段

| 阶段 | 内容 | 预估代码量 |
|------|------|-----------|
| Phase 1 | `AgentRole` 预设角色 (planner/executor/reviewer/researcher) + 增强 `AgentTool` | ~120 行 |
| Phase 2 | `Orchestrator` 任务分解 + DAG 调度 + 结果合成 | ~150 行 |
| Phase 3 | 协作模式 (pipeline/parallel/debate/verify) | ~100 行 |
| Phase 4 | `WorktreeManager` + 异步 Agent + Agent 间通信 | ~120 行 |

---

## 6. 优化方向五：权限与安全审查

> **可行性**: ⭐⭐⭐⭐ 高 | **当前状态**: 仅 Bash 正则黑名单 + 会话路径防护
>
> **目标**: 构建多层安全审查链路，替换简单黑名单为可配置的权限系统

### 6.1 当前状态分析

安全机制非常有限：

| 机制 | 文件 | 说明 |
|------|------|------|
| Bash 危险命令黑名单 | [bash.py](corecoder/tools/bash.py) 22-36 行 | 10 个正则，硬编码在工具代码中 |
| 会话路径防护 | [session.py](corecoder/session.py) | 两层路径遍历防御 |
| 实例级工具隔离 | [agent.py](corecoder/agent.py) 32 行 | `_tool_by_name` 字典 |

**缺失**：
- 无权限等级概念（只读/写入/Shell/网络）
- 无用户确认提示（所有工具直接执行）
- 无审计日志（不知道 Agent 做了什么操作）
- 无沙箱策略（文件系统和网络访问无限制）
- 黑名单硬编码在工具代码中，无法按项目/用户自定义

### 6.2 架构设计

```
corecoder/security/
├── __init__.py          # Guard + PermissionManager + 公开 API
├── permissions.py       # PermissionManager / PermissionRule / PermissionLevel
├── gate.py              # Guard: 多层审查链 + 输出脱敏
└── audit.py             # AuditLogger: JSONL 审计日志 + 自动轮转
```

### 6.3 多层审查链

```
工具调用请求
    │
    ▼
[Layer 1: 静态规则引擎]  → deny? → 阻止执行 + 记录审计
    │ 匹配权限规则 (allow/deny/ask)
    │ 检查频率限制 (calls/minute)
    │
    ▼
[Layer 2: 权限策略匹配]  → ask?  → 需要用户确认
    │
    ▼
[Layer 3: 用户确认交互]  → deny? → 阻止执行 + 记录审计
    │ (CLI 通过 rich 展示操作详情, y/n 确认)
    │
    ▼
[Layer 4: 审计日志记录]  → 所有操作记录到 JSONL
    │
    ▼
[Layer 5: 输出脱敏]      → 检查结果中的 API Key/Token/密钥 → 替换为 [REDACTED]
    │
    ▼
返回结果
```

### 6.4 核心数据结构

```python
PermissionLevel = Literal[
    "read",           # 只读: read_file, glob, grep
    "write_local",    # 项目目录内写入
    "write_global",   # 任意路径写入
    "shell_safe",     # 安全 Shell: ls, git status, ...
    "shell_any",      # 任意 Shell 命令
    "network",        # 网络访问
]

@dataclass
class PermissionRule:
    tool_name: str              # "bash" | "write_file" | "*" (通配)
    pattern: str                # 正则匹配参数
    action: Literal["allow", "deny", "ask"]
    reason: str                 # 决策原因说明
    level: PermissionLevel
    paths: list[str] | None     # 路径白名单 (glob 模式)
    require_confirm: bool       # 是否弹出确认提示
    priority: int = 0           # 优先级 (越高越先检查)
    max_frequency: int | None   # 频率限制 (calls/minute)

@dataclass
class SecurityDecision:
    allowed: bool
    needs_confirmation: bool
    reason: str
    rule: PermissionRule | None
```

### 6.5 权限配置体系

权限来源（优先级从高到低）：

```
1. 运行时用户确认     (最高优先级，当次有效)
2. ~/.corecoder/permissions.json   (用户级规则，跨项目)
3. .corecoder/permissions.json     (项目级规则，团队共享)
4. 内置默认规则       (安全默认值，保留现有 10 个危险模式 + 新增读工具允许)
```

**内置默认规则**：

| 规则 | 动作 | 说明 |
|------|------|------|
| Bash 危险模式 (10 个) | deny | 保留现有的 `_DANGEROUS_PATTERNS` 所有规则 |
| 安全 Bash 命令 (`ls`, `cat`, `git status`...) | allow | 免确认 |
| 读工具 (`read_file`, `grep`, `glob`) | allow | 永远安全 |
| 写工具 (`write_file`, `edit_file`) | allow | 项目目录内允许 |
| 子 Agent (`agent`) | allow | 允许生成 |

### 6.6 输出脱敏

```python
_SECRET_PATTERNS = [
    (r"sk-[a-zA-Z0-9]{20,}",              "[OPENAI KEY REDACTED]"),
    (r"AKIA[0-9A-Z]{16}",                 "[AWS KEY REDACTED]"),
    (r"eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]+", "[JWT REDACTED]"),
    (r"-----BEGIN .*PRIVATE KEY-----",    "[PRIVATE KEY REDACTED]"),
    (r"(?:api[_-]?key|token|secret|password)\s*[:=]\s*[\'\"][^\'\"]+[\'\"]",
                                           "[SECRET REDACTED]"),
]
```

### 6.7 集成方式

**Agent 层**：`_exec_tool()` 执行前经过 Guard：

```python
def _exec_tool(self, tc) -> str:
    if self.guard:
        decision = self.guard.review(tc.name, tc.arguments)
        if not decision.allowed:
            return f"[Security] Blocked: {decision.reason}"
        if decision.needs_confirmation:
            if not self.guard.confirm(decision):
                return f"[Security] User denied: {decision.reason}"
    # ... 执行 + guard.sanitize(result)
```

**BashTool**: `__init__` 接受可选的 `Guard` 对象，替代硬编码的 `_DANGEROUS_PATTERNS`。

**CLI 层**：
- 新增 `/permissions` 命令：查看权限规则列表
- 新增 `/permit <tool> <pattern>` 命令：添加允许规则
- 新增 `/deny <tool> <pattern>` 命令：添加拒绝规则
- 新增 `/audit` 命令：查看今日审计日志

### 6.8 审计日志

```
~/.corecoder/audit/
├── audit_2024-07-25.jsonl    # JSONL 格式，每行一条记录
├── audit_2024-07-26.jsonl    # 按日期分文件
└── ...                        # 自动轮转 (保留 30 天)
```

### 6.9 实施阶段

| 阶段 | 内容 | 预估代码量 |
|------|------|-----------|
| Phase 1 | `PermissionRule` + `PermissionManager` + 内置默认规则 + JSON 保存/加载 | ~150 行 |
| Phase 2 | `Guard` 多层审查链 + 输出脱敏 + 频率限制 | ~100 行 |
| Phase 3 | `AuditLogger` (JSONL + 自动轮转) + CLI 用户确认交互 | ~100 行 |
| Phase 4 | `SandboxPolicy` (文件系统/网络限制) + `ContentReviewer` (恶意代码检测) | ~100 行 |

---

## 7. 文件变更汇总

### 7.1 新增文件

```
corecoder/skills/__init__.py
corecoder/skills/base.py
corecoder/skills/registry.py
corecoder/skills/router.py
corecoder/skills/loader.py
corecoder/skills/builtins/__init__.py
corecoder/skills/builtins/code_review.py
corecoder/skills/builtins/test_gen.py
corecoder/skills/builtins/refactor.py

corecoder/memory/__init__.py
corecoder/memory/models.py
corecoder/memory/store.py
corecoder/memory/index.py
corecoder/memory/extractor.py
corecoder/memory/retriever.py
corecoder/memory/engine.py

corecoder/collab/__init__.py
corecoder/collab/orchestrator.py
corecoder/collab/roles.py
corecoder/collab/patterns.py
corecoder/collab/worktree.py
corecoder/collab/task.py

corecoder/security/__init__.py
corecoder/security/permissions.py
corecoder/security/gate.py
corecoder/security/audit.py
```

**新增文件总计**: 25 个

### 7.2 修改文件

| 文件 | 改动 | 兼容性 |
|------|------|--------|
| [agent.py](corecoder/agent.py) | 构造函数新增 3 个可选参数；`chat()` 注入 Skill/Memory 上下文；`_exec_tool()` 经过 Guard | ✅ 不传新参数行为不变 |
| [cli.py](corecoder/cli.py) | 新增 5 个 slash 命令；`quit`/`/save` 触发记忆提取 | ✅ 新增命令不影响现有命令 |
| [prompt.py](corecoder/prompt.py) | 支持外部上下文注入（Skill/Memory 片段） | ✅ 不注入时输出不变 |
| [session.py](corecoder/session.py) | `save_session` 后触发 `MemoryEngine.on_session_end` | ✅ 不配置 Memory 时无操作 |
| [config.py](corecoder/config.py) | 新增 `CORECODER_SKILLS_DIR`、`CORECODER_MEMORY_DIR` 等可选配置 | ✅ 不设置时用默认值 |
| [tools/bash.py](corecoder/tools/bash.py) | `__init__` 可选接受 Guard，替代静态 `_DANGEROUS_PATTERNS` | ✅ 不传 Guard 时用默认正则 |
| [tools/agent.py](corecoder/tools/agent.py) | 支持 `role`, `async_mode`, `worktree`, `tools` 参数 | ✅ 不传新参数行为不变 |
| [tools/\_\_init\_\_.py](corecoder/tools/__init__.py) | 注册新工具（如有） | ✅ 仅追加 |
| [\_\_init\_\_.py](corecoder/__init__.py) | 导出新模块 + 可选 import | ✅ 原有导出不变 |

### 7.3 代码量预估

| 模块 | 新增文件 | 预估代码量 | 新增依赖 |
|------|---------|-----------|---------|
| Skills | 9 个 | ~360 行 | 无 |
| Memory | 7 个 | ~410 行 | 无 |
| Context | 0 (修改 1) | ~230 行 | tiktoken (可选) |
| Multi-Agent | 6 个 (+1 修改) | ~490 行 | Git (系统工具) |
| Security | 4 个 | ~450 行 | 无 |
| **总计** | **26 个文件** | **~1,940 行** | **tiktoken (可选)** |

引擎代码量从 ~1,081 行扩展到约 3,000 行——仍然在一个下午可通读的范围。

---

## 8. 测试策略

### 8.1 新增测试文件

```
tests/test_skills.py         # Skill 匹配、路由、注册、依赖解析
tests/test_memory.py         # Memory CRUD、提取、检索、索引
tests/test_context_enh.py    # 精确 Token 计数、语义截断、增量摘要
tests/test_collab.py         # 角色、任务分解、编排器、Worktree
tests/test_security.py       # 权限规则、Guard 链、审计日志、脱敏
```

### 8.2 测试模式（复用现有模式）

- **pytest fixtures**: `tmp_path`, `monkeypatch` — 与现有测试一致
- **直接函数测试**: 导入并调用 → 断言返回值
- **Mock 替换**: Mock LLM 返回固定结果，测试提取器/编排器/检索器
- **不变量保护**: 验证 Guard 不改变工具执行结果（除脱敏外）
- **参数化测试**: 多场景覆盖（不同输入 → 不同 Skill 匹配结果等）
- **线程安全测试**: Worktree 隔离的并发安全性

### 8.3 兼容性保证

- 现有 86 个测试保持全绿
- 不传新参数的 `Agent()` 行为与 v0.4.0 完全一致
- CI 矩阵 (3 OS × 4 Python) 依然通过

---

## 9. 验证方案

### 9.1 单元测试

- 每个新模块配套测试文件，目标代码覆盖率 ≥ 现有水平
- 关键路径必须有参数化测试覆盖

### 9.2 集成测试

| 场景 | 验证目标 |
|------|---------|
| Skill 路由 | 用户输入 → 自动匹配 Skill → 注入提示词 → 执行 → 返回结果 |
| 记忆闭环 | 会话 A 生成记忆 → 会话 B 首条消息 → 相关记忆被检索 → 注入系统提示词 → Agent 遵循 |
| 多 Agent | Orchestrator 分解任务 → 按 DAG 并行执行 → 结果合成 → 输出一致 |
| 安全审查 | 危险命令 → Guard 拦截 → 审计记录 → 输出脱敏 |

### 9.3 手动验证

- REPL 交互：`/skills` `/memory` `/permissions` `/audit` `/team` 等新命令功能正常
- 跨会话记忆：连续两次对话，第二次 Agent 体现出第一次学到的偏好
- 权限系统：`/permit bash "git commit"` → 下次 `git commit` 免确认执行
- 审计日志：`/audit` 查看操作记录，确认危险操作被正确记录

### 9.4 回归验证

```bash
# 现有测试必须全部通过
python -m pytest tests/ -v

# 编译检查
python -m compileall corecoder/

# 代码风格
ruff check corecoder/
```
