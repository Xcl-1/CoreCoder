基于我们所有讨论的共识——坚守 ReAct 手写循环、采纳 Harness 工程思想、拒绝 LangGraph 式编排绑架——我为你梳理了一份极简的 CoreCoder 进化路线图。

这份规划遵循 “先保命（可观测），再强身（工程化），后出拳（智能化）” 的逻辑，分为三个大版本里程碑：

🧱 里程碑 1：v0.2 —— “看得见”（可观测性底座）
（目标：把“黑盒”变成“白盒”，代码量 1k → 3k）

这是绝对不能跳过的阶段。 没有它，后面所有的并发和多 Agent 都会变成无法调试的“赛博鬼打墙”。

核心动作：实现 “回放日志（Replay Log）”。将每一次 思考 -> 行动 -> 观察 的完整上下文、工具调用参数、返回结果全部序列化为 JSON 落盘。

工程改造：引入 Pydantic 重写所有数据结构（State, Message, ToolCall），利用 asyncio 改造文件读写和 subprocess，为并行执行铺路。

验收标准：当 Agent 抽风改错代码时，你能打开日志文件，精确查到“第 3 步它为什么要删掉那个函数”。

⚙️ 里程碑 2：v0.5 —— “站得稳”（工程健壮性 & 规划能力）
（目标：从“莽撞执行”进化为“三思后行”，代码量 3k → 8k）

核心动作：

引入 Plan 模式：在主循环中增加 planning 状态。遇到复杂任务（如“重构整个登录模块”），强制 LLM 先输出结构化的 plan.json，你确认后再进入执行循环。（这正是 Claude Code 的核心交互体验）
引入沙箱（Sandbox）：将工具执行（尤其是文件写入和 Shell 命令）挂载到 Docker 或虚拟环境中，禁止 Agent 访问系统敏感目录。
引入 tree-sitter：替换正则表达式，实现基于 AST（抽象语法树）的精准代码编辑，确保改完不会破坏括号匹配。
技术选型：此时可接入 langchain-openai 作为底层 API 客户端，但严禁引入 langchain.agents。

🚀 里程碑 3：v1.0 —— “跑得快”（高级自治 & 多 Agent 委派）
（目标：具备大型代码库的自主修复能力，代码量 8k → 15k+）

核心动作：

并行工具执行：利用 Milestone 1 改造的 asyncio 基础，让 Agent 在一次推理中同时读取 5 个文件，而不是排队读，大幅降低延迟。
实现 Spawn 委派（多 Agent）：绝对不用 LangGraph。而是在工具列表里新增一个 spawn_sub_agent(task_desc) 函数。主 Agent 遇到跨模块任务时，直接 fork 一个带独立上下文的新 Agent 实例去跑，跑完回来汇报摘要。
上下文三层压缩：强化现有的压缩策略，当对话超过 128k 时，自动将早期的“工具调用细节”压缩成“语义摘要”，保住关键指令不丢失。





















































CoreCoder 下一步开发建议
项目现状
CoreCoder v0.4.0 已经从最初的 1,081 行引擎代码发展到约 1,700+ 行，近期密集引入了多个 v1.0 特性：三级上下文压缩（tiktoken + 类型感知截断 + 增量摘要）、多角色 Agent 委派（planner/executor/reviewer/researcher）、AST 精准编辑、Plan 模式、Sandbox 沙箱、异步并行工具执行（含读写依赖调度）、Replay 日志。101/102 个测试通过。

OPTIMIZATION_PLAN.md 规划的五大方向中，上下文压缩和多 Agent 协作已部分落地，但 Skill 系统、记忆系统、权限安全系统 三个方向完全空白。

⚠️ 立即处理：轮换 API Key
代码审查发现 .env 文件中包含真实的 DeepSeek API Key。虽然 .env 在 .gitignore 中，但该 Key 已被 AI 工具读取。建议立即到 DeepSeek 控制台轮换该 Key。

阶段一：稳住已有功能 + 还技术债（优先，1-2 周）
快速修复（半天）：

修复 test_config_defaults 1 个失败用例
消除 _unified_diff 函数在 edit.py 和 edit_ast.py 中的完全重复定义，提取到共享工具模块
tiktoken 已在 context.py 中使用但未在 pyproject.toml 声明为可选依赖
依赖版本加上上界（openai>=1.0 → openai>=1.0,<3.0）
config.py 补参数校验（max_tokens > 0、temperature 范围、provider 合法值）
测试补充（3-5 天）： Plan 模式、edit_ast、Sandbox、Replay 日志、Role 系统、并行调度、context.py L2 增量摘要，这些新特性均无测试覆盖。

代码质量改善（1-2 天）：

引入 logging 模块替代 print() 调试输出
收紧 18 处 except Exception 为更具体的异常类型
6 个工具重复的 async execute() → asyncio.to_thread(self._execute_sync) 模式提取到 Tool 基类
补 agent.py 中 on_token/on_tool 等回调的类型标注
阶段二：权限与安全系统（~450 行，1-2 周）
当前安全机制仅 Bash 正则黑名单 + 路径白名单，是 README 明确指出的薄弱点。自包含在新建的 corecoder/security/ 包：

PermissionManager：六级权限（read → shell_any），用户级/项目级 JSON 配置，allow/deny/ask 三种动作
Guard：五层审查链（静态规则 → 策略匹配 → 用户确认 → 审计记录 → 输出脱敏）
AuditLogger：JSONL 格式，按日期分文件，30 天自动轮转
CLI 新增 /permissions、/permit、/deny、/audit 命令
阶段三：自进化记忆系统（~410 行，1-2 周）
让 Agent 具备跨会话学习能力——记住用户偏好、项目约定、历史反馈。自包含在 corecoder/memory/ 包：

文件级存储：每条记忆一个 Markdown，MEMORY.md 做索引
LLM 驱动提取：会话结束 → 自动分析对话 → 提取关键信息 → 去重合并
关键词检索：新会话开始 → 匹配相关记忆 → 注入系统提示词
阶段四：Skill 能力体系（~360 行，1-2 周）
建立在已有 Role 系统之上，让 Agent 行为定制从"改源码"变为"加载 Skill 文件"。Skill 不直接调用工具，而是注入专属系统提示词片段：

意图路由：正则 triggers + 关键词累积计分 → confidence 排序
3 个内置 Skill：code-review、test-gen、refactor
用户自定义：~/.corecoder/skills/*.skill.md 自动加载
后续可选方向
Multi-Agent Orchestrator：将已有的 spawn() 升级为完整的编排器 + 四种协作模式 + Worktree 隔离
MCP 协议集成：接入外部工具生态
文档更新：article/ 系列文章反映 v1.0 新增特性
设计原则
每个模块独立、可读、可 fork，引擎总代码量控制在 ~3,000 行以内
向后兼容（不传新参数的 Agent() 行为不变）
无外部数据库，延续文件级存储
一个模块一个 PR，各阶段独立交付






























结合现有 Agent + Role + Session + Prompt 架构，建议按下面两阶段实施。此次仅规划，没有修改代码。
阶段三：自进化记忆系统
目标：完成“会话提取 → 文件存储 → 相关检索 → 新会话注入”的闭环。
1. 建立 Memory 模块
在 corecoder/memory/ 下新增：
- models.py：记忆类型及数据模型
- store.py：Markdown 记忆的增删改查
- index.py：生成和维护 MEMORY.md
- extractor.py：调用 LLM 分析会话
- retriever.py：关键词匹配和排序
- engine.py：统一编排生命周期
- __init__.py：公开接口
存储位置默认为 ~/.corecoder/memory/，每条记忆一个 Markdown 文件。
2. 实现提取和合并
会话结束时：
1. 过滤工具输出、临时信息和敏感数据。
2. 让 LLM 提取用户偏好、项目约定和明确反馈。
3. 与现有候选记忆比较。
4. 输出 create / merge / ignore 操作。
5. 更新 Markdown 文件并重建索引。
需要保证重复关闭或重复提取不会产生多份相同记忆。
3. 实现新会话检索
首次收到用户消息时：
- 按标题、关键词、类型和项目范围计算相关度。
- 返回 Top-K 记忆。
- 作为临时系统提示词片段注入。
- 不写入 agent.messages，避免污染会话存档和上下文压缩。
用户偏好可全局生效，项目约定只在对应项目目录下生效。
4. 接入生命周期
主要修改：
- [agent.py](D:/develop/Project/Python/CoreCoder/corecoder/agent.py)：接入检索和会话结束提取
- [cli.py](D:/develop/Project/Python/CoreCoder/corecoder/cli.py)：确保交互模式、单次模式和 /reset 都正确结束会话
- [config.py](D:/develop/Project/Python/CoreCoder/corecoder/config.py)：增加启用开关、目录和 Top-K 配置
预计 1–2 周，Memory 包约 410 行，不包含测试和少量集成代码。
阶段四：Skill 能力体系
目标：通过加载 Skill 提示词定制 Agent 行为，不改变工具执行机制。
1. 建立 Skill 模块
在 corecoder/skills/ 下新增：
- base.py：Skill、SkillMatch
- registry.py：注册、覆盖和查询
- router.py：意图计分与排序
- loader.py：加载用户 Skill 文件
- builtins/：三个内置 Skill
- __init__.py：公开接口
2. 实现意图路由
每次收到新的用户请求时：
- 正则 trigger 命中获得较高分。
- 多个关键词命中后累积加分。
- 过滤低置信度结果。
- 按 confidence 排序，最多激活 1–2 个 Skill。
Skill 只提供提示词片段，不调用、增加或限制工具。
3. 提供三个内置 Skill
- code-review：正确性、安全性、边界情况和可维护性审查
- test-gen：分析现有测试风格，补充正常、异常和边界用例
- refactor：保持行为不变，小步修改并验证测试
4. 加载用户 Skill
启动时扫描：
~/.corecoder/skills/*.skill.md
文件头保存名称、描述、triggers、keywords 和优先级，正文作为提示词片段。格式错误的文件跳过并记录警告；同名用户 Skill 可以覆盖内置 Skill。
5. 与 Role 体系组合
统一系统提示词组装顺序：
基础提示词
→ 相关记忆
→ 激活的 Skill
→ Role 约束
Role 权限和工具限制优先于 Skill，确保 Reviewer/Researcher 不会因为 Skill 而获得写权限。
预计 1–2 周，Skills 包约 360 行。
验收重点
- 会话 A 产生记忆，会话 B 能自动检索并遵循。
- 相同信息重复出现时更新原记忆，不创建副本。
- 项目记忆不会泄漏到其他项目。
- Skill 路由结果稳定、置信度排序正确。
- 自定义 .skill.md 无需修改源码即可生效。
- Memory 或 Skill 加载失败不能阻塞正常聊天。
- 现有 Role、Session、安全系统和全部测试保持兼容。
建议实施顺序为：先完成 Memory 的存储与检索，再接入 LLM 提取；随后完成 Skill 路由和内置 Skill，最后统一整理系统提示词组装链路。