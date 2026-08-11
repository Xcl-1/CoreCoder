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