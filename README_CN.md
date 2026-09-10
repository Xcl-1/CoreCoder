<div align="center">

# CoreCoder

**编程 agent 里的 nanoGPT。1081 行纯 Python，读懂一个 coding agent 到底怎么运作，再 fork 出你自己的。**

*learn from it · fork it · ship something better*

中文 | [English](README.md) | [配套源码导读 · 八篇双语](article/)

[![PyPI](https://img.shields.io/pypi/v/corecoder)](https://pypi.org/project/corecoder/)
[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://github.com/he-yufeng/CoreCoder/actions/workflows/ci.yml/badge.svg)](https://github.com/he-yufeng/CoreCoder/actions)
[![engine](https://img.shields.io/badge/engine-1081_LoC-blue)](article/)
[![源码导读](https://img.shields.io/badge/源码导读-8篇双语-orange)](article/)

</div>

- **读得完。** 一个下午读完整个引擎，没有一处藏着你看不懂的魔法。
- **改得动。** 每一行都能在你自己机器上下断点、改了再跑。它真能干活，所以这份参考是活的，不是示意图。
- **留白即起点。** 刻意只留最小核心，没做的那些不是半成品，是留给你 fork 出更好东西的地方。

## 和谁比

| | CoreCoder | Claude Code | aider | nanoGPT |
|---|---|---|---|---|
| 代码量 | 引擎约 1081 行 / 整包 1714 行 | 几十万行（闭源） | 数万行 Python | 约 600 行（两个文件） |
| 读完要多久 | 一个下午 | 读不了（闭源） | 得啃几天 | 一个下午 |
| 能不能下断点改了再跑 | 能，每一行 | 不能 | 能，但量大 | 能 |
| 定位 | 读懂并 fork 出你自己的 agent | 生产级编程助手 | 终端结对编程 | 教学用最小 GPT |

nanoGPT 那一列是拿来对照的：它最小、可读，但教的是训一个 GPT。CoreCoder 想干的是同一件事，只是把对象换成一个能真正改代码的 agent。和 Claude Code、aider 摆在一起，不是要跟它们抢用户，CoreCoder 是借它们来学、来起步的那块地基，根本不在一个赛道。

## 这是什么

我一直觉得 coding agent 被讲得太玄了。把 Claude Code、Cursor 这类工具扒到底，核心是一个 while 循环套着一个大模型，外加七八个让它能真正动手的工具。难的从来不是这个循环，而是循环跑进真实世界以后要兜的那些底。CoreCoder 就是把这个核心老老实实写出来的最小版本。

引擎部分（循环、模型接口、上下文、工具、会话）去掉空行和注释是 1081 行。连最外层的 CLI、配置、打包一起算，整个包 18 个文件、物理 1714 行、净 1385 行，每个文件都短到能一口气读完。

它真能跑：读写文件、执行 shell、派子 agent、分三层压上下文，还能随时把这趟烧掉的 token 和美元数报给你，86 个测试是绿的。但能跑不是为了劝你拿去日用，而是为了让这份「注释」不撒谎：一个解释 agent 怎么运作的范例，自己得真能运作。

代码来自一次公开拆解。公开的源码分析里，Claude Code 这类生产级 agent 暴露出不少关键架构，我挑出最核心的一层，用尽量少的代码诚实地复写了一遍。所以读 CoreCoder，约等于读一份基于公开源码分析的「可运行注释版」：讲的是这类 agent 的核心思路，而它本身只是最小复写，就摆在你机器上，随你拆、随你改。

<p align="center">
  <img src="assets/demo.png" width="820"
       alt="CoreCoder 一次真实运行：corecoder -p 让它修 buggy.py，agent 自己读文件、改代码、跑验证、给出结论">
</p>

<p align="center"><sub><i>这一千行真能跑通一个完整回合：让它修 buggy.py，它自己读文件、改代码、跑一遍确认、再给结论。看完就回来读代码。</i></sub></p>

这份 README 也就按这条线铺开：上半带你**读懂**（代码地图、主循环、八篇导读），下半带你 **fork** 它、再指几个能往更好里做的方向。

## 先跑一次（读之前的五分钟）

读源码之前，先让它在你机器上活一次，建立点体感。它是个拿来 fork 的地基，所以推荐直接 clone 下来、可编辑安装，边读边改：

```bash
git clone https://github.com/he-yufeng/CoreCoder
cd CoreCoder
pip install -e .
```

只想先跑起来找找感觉，直接 `pip install corecoder` 也行。

给它一个模型加一把 key 就能动。默认走 OpenAI 兼容接口，换 provider 通常只是改两个环境变量：

| Provider | 环境变量示例 |
|---|---|
| OpenAI（默认 `gpt-5.5`） | `OPENAI_API_KEY=sk-...` |
| DeepSeek | `OPENAI_API_KEY=sk-... OPENAI_BASE_URL=https://api.deepseek.com CORECODER_MODEL=deepseek-chat` |
| 本地 Ollama | `OPENAI_API_KEY=ollama OPENAI_BASE_URL=http://localhost:11434/v1 CORECODER_MODEL=qwen2.5-coder` |

Kimi、Qwen 这些同样是改这两个变量；连 OpenAI 兼容接口都不给的 provider，装上可选的 LiteLLM 后端（`pip install "corecoder[litellm]"`）能路由一百多家。第三篇文章把这块讲得更细。key 可以直接 `export`，也可以在项目根目录扔个 `.env`，启动时自动加载。然后：

```bash
corecoder                                  # 交互式 REPL
corecoder -p "给 parse_config() 加错误处理"   # 一次性模式，干完就退
```

## 读懂它：代码地图

整个项目摊开就这么大，clone 之前扫一眼，心里就有数了。这也是它和 Claude Code 几十万行最实在的区别：你能把它当一本书的目录来读。建议从 `agent.py` 的主循环读起，那是整个 agent 的心脏。

```
corecoder/
├── agent.py        agent 主循环 + 并行工具执行       150 行   ← 从这里开始读
├── llm.py          流式客户端 + 重试 + 成本统计       336 行
├── context.py      三层上下文压缩                     210 行
├── session.py      会话存盘 / 续聊 + 路径穿越防护      97 行
├── prompt.py       系统提示词                          33 行
├── cli.py          REPL + 斜杠命令 + 一次性模式        270 行
├── config.py       环境变量配置                        57 行
└── tools/
    ├── bash.py       shell + 危险命令闸 + cd 追踪      127 行
    ├── edit.py       唯一匹配搜索替换 + diff            92 行
    ├── grep.py       内容搜索                           79 行
    ├── glob_tool.py  文件名匹配                         47 行
    ├── read.py       文件读取                           53 行
    ├── write.py      文件写入                           38 行
    ├── agent.py      子 agent 派生                      58 行
    └── base.py       工具基类                           27 行
```

七个工具：`bash`、`read_file`、`write_file`、`edit_file`、`glob`、`grep`、`agent`（派子 agent）。其余都是包在引擎核心外面的 CLI 外壳、配置和打包。

## 一个 while 循环就是 agent 的本体

一个 agent 的本体，一句话就能讲清：把用户的话交给模型，模型想调工具就执行，把结果塞回上下文，再问模型，直到它不再要工具、给出回答。落到代码，也就十来行：

```python
# corecoder/agent.py · 主循环（精简骨架）
def chat(self, user_input):
    self.messages.append(user_input)

    for _ in range(self.max_rounds):                   # 循环有上限，跑不飞
        reply = self.llm.chat(self.messages, self.tools)   # 交给模型规划下一步
        if not reply.tool_calls:                       # 模型不再要工具
            return reply.text                          #   → 收工，把回答给用户
        results = run_parallel(reply.tool_calls)       # 要工具就并发执行
        self.messages += results                       # 结果回灌，进入下一轮

    return "(已达轮次上限)"
```

就这么点。这个循环的核心骨架就二十来行，把并行执行和被 Ctrl+C 打断后的回填都算上，也才四十多行。CoreCoder 一千多行里剩下的，几乎全在收拾它真跑起来之后冒出来的岔子。`llm.py` 最后成了全项目最大的文件，不是因为调模型有多难，而是流式返回里一个工具调用的参数会被切成好几段先后送到、得按顺序拼回去，provider 偶尔吐半截 JSON 或把 usage 填成 null，限流（429）、超时、连接中断和 5xx 都得退避重试，其余 4xx 该直接抛就别硬试。这些不起眼的脏活，而不是那个循环，才是一个 agent 从能演示走到能交付真正吃工程功夫的地方；第三篇文章顺着它拆到每一行。

有三个决定值得单独看，因为它们是「先读懂别人怎么做」之后才做得出的取舍，也是你 fork 自己 agent 时可以直接抄走的判断。

**`edit_file` 用唯一匹配的搜索替换，不靠行号。** 行号这东西，模型只要数偏一行，就会悄悄改错地方；锚定一段唯一的原文：匹配不到，就把文件开头甩回去让模型照着重新锚定；匹配到多处，就让它多带几行上下文再来，而不是赌一个。改成功了，连一段 diff 一起返回。失败能复位、成功能复核，闭环都收在工具自己手里。

**上下文不是满了才一刀切，而是按代价从轻到重分三档退让。** 先在半满（50%）时把超长的工具输出就地截短，这一档纯机械、不花一次模型调用；到 70% 还压不下去，就把较早的轮次交给模型总结成一段摘要，最近几轮原文原样留着；逼到 90% 才进应急档，连摘要带最近几轮一起收到最紧。粗暴截断往往恰好丢掉一个长任务最依赖的早期决定；分层退让，是让它按重要性从低到高一档档地让，而不是一上来就把最老的决定整段切掉。

超大工具结果会在第一次进入历史前外置化。对话只保留稳定的摘要预览和 `artifact://sha256/...` 引用；只读工具 `retrieve_context` 可以按关键词或行范围恢复原始证据。达到摘要水位后，旧对话会变成经过 Schema 校验的 JSON 检查点，明确记录当前目标、约束、决策、文件、验证结果、错误、待办和 artifact 引用。检查点带版本号，只有回落到低水位或积累了足量新消息后才允许再次更新，使两次压缩之间的 Prompt Cache 前缀保持稳定。设置 `CORECODER_CONTEXT_ARTIFACTS=0` 可关闭外置存储；`CORECODER_CONTEXT_ARTIFACTS_DIR` 和 `CORECODER_CONTEXT_ARTIFACT_THRESHOLD` 分别控制存储目录与默认 12,000 字符阈值。`CORECODER_CONTEXT_ARTIFACT_TTL_DAYS`（默认 30 天）和 `CORECODER_CONTEXT_ARTIFACT_MAX_MB`（默认 256 MB）限制保留周期与总容量；系统先清理过期项，再按最旧优先释放容量。`/tokens` 会在 Provider 支持时显示 Prompt Cache 命中/未命中用量，并显示外置、检索、清理、压缩和检查点指标。

**子 agent 现在统一运行在主 agent 持有的控制平面之后。** 模型可见接口不再提供 `single`/`coding_team` 开关：简单任务由主 agent 自己完成；存在独立子任务时，主 agent 根据实际情况发出零个、一个或多个 `agent` 调用。同一次响应中的调用会在硬上限内并行，有依赖的实现和审查必须等前置结果返回后再委派。每次委派都先构造成经过校验的 `TaskSpec`：目标、最小上下文、精确工具白名单、读写根路径、Token 与工具调用预算、超时、角色、工作区模式和验收条件。`TaskController` 统一管理状态机、并发上限、超时和取消。子 agent 使用独立历史，拿不到 `agent` 工具，不能申请权限提升，只能返回有长度边界的 `TaskResult`；完成状态、实际修改文件、用量和越权次数由运行时生成，不接受子 agent 自报。最终复核与验收始终由主 agent 完成。

文件/搜索工具的相对路径和 Shell 命令都会以各 Agent 的 `workspace_root` 为基准。委派任务的读写根路径会在执行前完成规范化，并且规范化后仍必须位于父工作区内；越界范围会在子模型运行前直接拒绝。

作为库使用时，可以直接提交协议对象：

```python
from corecoder import Agent, TaskRole, TaskSpec

spec = TaskSpec(
    objective="检查认证实现并找出入口",
    role=TaskRole.RESEARCHER,
    allowed_tools=("read_file", "grep", "glob"),
    read_paths=("corecoder/security",),
    token_budget=8_000,
    timeout_seconds=120,
    acceptance_criteria=("引用定义所在的文件",),
)
result = await agent.delegate(spec)
assert result.requires_parent_review
# 主 agent 独立核验每项条件后：
# accepted = agent.accept_task(result.task_id, parent_verified_checks)
```

CoreCoder 现在只向用户提供统一的 LangGraph 编排路径。Native 适配器仅作为内部兼容和评估基线保留：

```bash
pip install -e .
corecoder
```

```python
from corecoder import Agent, TaskSpec, WorkflowRequest

agent = Agent(llm=llm)
workflow = await agent.run_workflow(WorkflowRequest(
    task=TaskSpec(objective="实现并验证受限范围内的改动"),
    max_replans=1,
    max_total_tokens=24_000,
))
```

工作流固定为 `规划 -> 审批 -> 执行 -> 验证 -> 审查 -> 决策`。图本身拿不到工具或文件系统，实际执行仍然经过 `TaskController`；规划器可以细化指令，但不能扩大角色、工具、路径、预算、超时、验收条件或执行模式。策略违规和预算耗尽不会重试，写入型重试必须使用 worktree 隔离。无路径约束的高风险工具会在执行前中断，只有与当前任务摘要严格绑定的审批决定才能恢复。

公共 Agent API 不再存在 Native 执行旁路。主 Agent 与子 Agent 的每一次 `chat()` 都经过 LangGraph turn 生命周期（`规划 -> 执行 -> 验证 -> 审查`）；前台委派、后台和 Durable 任务、批量委派以及团队阶段还会经过上面的受限任务图。原始模型/工具循环和控制器直接执行器只作为图节点内部回调，从而既避免递归，又保持唯一的用户执行路径。`agent.last_turn_workflow` 可用于查看最近一次对话的真实阶段轨迹。

默认检查点只保存在内存中。需要进程重启后恢复时，库调用方可传入 `encrypted_sqlite_checkpointer(path, key=...)`；序列化类型采用严格白名单，AES 密钥只能是 16、24 或 32 字节。内置规划、验证和审查器刻意保持确定性：LangGraph 增强的是流程控制，不会自动提高模型智力。需要更强思考质量时，应注入相互独立的回调，并继续以父 Agent 实际运行的测试和验收结果为准。

`measure_workflow()` 与 `summarize_workflows()` 会从结构化结果中统计成功率、重试、Token、耗时、工具调用、策略违规、报告测试和父 Agent 验收指标，因此可以对 Native 与 LangGraph 做 A/B 对照，而不必相信模型自行书写的结论。

`agent.tasks.snapshot(task_id)`、`list_tasks()` 和 `events()` 提供有界且不含任务
Prompt 的控制平面状态查询。任务生命周期事件同时写入现有 JSONL 审计日志，包含任务、
父子 Agent、角色、权限范围和工作区身份。审计写入失败不会改变任务执行结果，内存中的
任务历史由 `task_history_limit` 限制容量。

长任务可以在不阻塞调用方的情况下提交：

```python
task_id = await agent.submit_task(spec)
snapshot = agent.tasks.snapshot(task_id)
result = await agent.wait_task(task_id, timeout=30)  # 只限制本次等待
# agent.cancel_task(task_id)                         # 显式取消后台任务

# 基于游标的进度只包含工具名和控制器里程碑，不包含参数。
batch = await agent.wait_task_events(task_id, after_sequence=0, timeout=30)
cursor = batch.next_sequence
```

等待方自身被取消或等待超时不会终止后台任务；只有控制平面的显式取消或任务自己的截止时间
会停止执行。
主模型也可以调用 `agent(background=true)` 获得相同行为，随后使用仅父 Agent 可用的
`task_control` 工具管理任务。交互式 CLI 提供 `/tasks`、`/task <id>`、
`/wait-task <id> [seconds]` 和 `/cancel-task <id>`；`/watch-task <id> [seconds]`
按游标输出实时进度。CLI 在等待终端输入时仍保持事件循环运行，因此后台任务会在两次用户
命令之间继续执行。进度事件只包含控制器里程碑和工具名，不包含工具参数、模型 Token 或
思维过程；若客户端游标落后于有界保留范围，返回结果会明确标记 `history_truncated`。

CLI 还会在 `~/.corecoder/tasks` 下维护有界的追加式任务日志；目录按租户和用户隔离，
文件名由工作区路径摘要生成。同一工作区的新进程可以通过 `/tasks`、`/task` 和
`/wait-task` 查看此前的终态结果。没有写入终态就退出的任务会被标记为 `interrupted`，
且绝不会自动重放。日志不保存原始 `TaskSpec.objective` 和上下文，但会保存终态
`TaskResult` 供主 Agent 复核。设置 `CORECODER_TASK_PERSISTENCE=0` 可关闭持久化，
`CORECODER_TASK_STATE_DIR` 可修改存储目录。

每个工作区还通过心跳租约保证同一时刻只有一个进程拥有调度和任务日志写入权。第二个进程会
以只读观察模式打开同一份持久化历史：可以刷新、等待和观察进度，但不能提交或取消子任务。
`/claim-tasks` 只会显式取得已释放或过期的租约，并在取得所有权后才把真正遗留的任务标记为
`interrupted`。同一主机上的存活 PID 不会仅因心跳过旧而被抢占；跨主机共享目录可通过
`CORECODER_TASK_LEASE_STALE_SECONDS` 调整过期阈值（默认 30 秒，最小 5 秒）。

持久执行必须显式开启。只有后台 `TaskSpec` 设置 `durable=True`（或调用
`agent(background=true, durable=true)`）时，完整任务规格才会在调度前进入经过 Fernet
认证加密的队列。CLI 取得工作区租约后会恢复有效队列项；正常终态和显式取消会删除队列项，
崩溃或关闭时未完成的工作则留给下一任所有者。被篡改、密钥不匹配、格式非法或超过容量的
条目绝不会执行。自动生成的队列密钥保存在租户/用户任务状态目录的 `.task-queue.key` 中，
并在系统支持时限制文件权限。通过模型工具启用 durable 会要求一次新的用户确认，因为这会
持久化 objective 和 context；普通任务继续只写不含 Prompt 的日志。

在 Worker 应负责的仓库目录中，可以启动独立的前台消费者：

```bash
corecoder worker                       # 持续轮询，默认每 1 秒一次
corecoder worker --poll-interval 0.25 # 自定义有界轮询间隔
corecoder worker --once               # 清空当前队列后退出
corecoder worker --workspace ../repo-a --workspace ../repo-b
corecoder worker --workspace ../repo-a --workspace ../repo-b --workspace-concurrency 2
```

Worker 与交互式 CLI 复用同一个 Agent、租约、控制器、工具边界、预算、Worktree 合并、
审计和加密队列。Worker 没有交互确认回调，因此任何需要新增人工批准的操作都会失败关闭；
durable 任务必须已由提交客户端批准。未取得租约或关闭了任务持久化时，Worker 会以非零状态
退出。Ctrl+C 会先标记关闭状态再让 asyncio 取消子任务，从而把未完成队列项留给下一任
Worker。Worker 持有租约期间，另一个 CLI 可以提交显式 durable 任务，但仍不能直接执行或
取消委派任务。重复传入 `--workspace` 会启动工作池，每个解析后的目录都有独立 Agent、工具
注册表、控制器、队列和租约。各工作区并发扫描，因此繁忙仓库不会阻塞其他仓库接纳任务。
租约同时承担跨进程分片：已被另一个存活 Worker 占用的工作区会被跳过，其余工作区继续运行；
重复路径和不存在的目录会在执行前被拒绝。
工作池的全局执行上限默认为 4，可通过 `--workspace-concurrency` 设置为 1–32；其下仍会应用
各工作区控制器自身的并发限制。
在 `--once` 模式下，两种 Worker 都会分别报告成功与失败任务数；只要存在终态任务失败或
Worker 内部错误，进程就会返回非零状态。

两种执行后端都走同一套协议。`fork` 在共享目录中执行。默认每次模型响应最多委派 4 个子 agent，控制器最多同时运行 3 个；可通过 `CORECODER_MAX_SUBAGENTS_PER_ROUND` 和 `CORECODER_TASK_CONCURRENCY` 调整为 1–32，并发数不得大于单轮委派上限。`worktree` 要求主 Git 工作区干净，在受管的 detached worktree 中运行子 agent，收集二进制差异，先由中央执行 `git apply --check`，再应用到主目录并纳入 `/undo`。发生冲突时主目录保持不变，隔离目录会留下供复核。`bash` 和 `undo_changes` 没有可可靠检查的文件路径参数，委派任务默认拒绝；只有库调用方明确选择无路径约束工具时才能开放，而 Worktree 任务始终拒绝它们。

模型不再选择固定团队模式，而是按任务实际需要显式委派最少数量的 researcher、executor 或 reviewer：简单请求可以不用子 agent，多个独立调查则可以动态展开。`Agent.run_team()` 仅作为库调用方主动选择固定流水线时使用的编程接口保留。只读任务可以选择最多三次共享总预算的重试；连续失败达到阈值后，控制器会先熔断，停止接纳更多子任务。

每一个「为什么」，下面的文章系列都拆到了具体代码行。

## 配套源码导读 · 八篇双语

我还写了一套双语源码导读，一篇导言加七篇正文，每篇都配英文镜像（`_EN.md`）。它对着 CoreCoder 的真实代码，讲 Claude Code 这类 agent 的内部构造。有一条给自己立的硬规矩：每一处行数、每一段代码都从仓库里现读现核，绝不凭印象编。前六篇带你读懂，第七篇带你 fork，哪篇先读都行。

- **[导言 · 用 CoreCoder 读懂 Claude Code，再造一个你自己的](article/00-index.md)**
- **[01 一个 agent 的本体，是一个 while 循环](article/01-the-loop.md)** — `agent.py` 的主循环、打断与轮次上限
- **[02 工具系统：让模型安全地动手](article/02-tools.md)** — `tools/` 七个工具与 bash 安全闸
- **[03 接入任意大模型，顺便把账算清楚](article/03-llm-and-cost.md)** — `llm.py` 的 provider 包装、重试与成本统计
- **[04 用有限的窗口扛住一个长任务](article/04-context.md)** — `context.py` 的三层压缩与孤儿 tool 消息
- **[05 并行执行与子 agent](article/05-parallel-and-subagents.md)** — 线程池并发与子 agent 隔离
- **[06 把它跑成一个真正的命令行工具](article/06-session-and-cli.md)** — `session.py` 与路径穿越防护
- **[07 Fork CoreCoder，搭一个你自己的 coding agent](article/07-build-your-own.md)** — 从 fork 到加自定义工具到换模型

## Fork 它，造个更好的

读懂之后，最自然的下一步就是 fork。起手不用伤筋动骨：

- **换个你常用的模型。** 就是上面那两个环境变量，`llm.py`（336 行）是所有 provider 适配的入口。
- **加一件你自己的工具。** 照 `tools/base.py`（27 行）的工具基类写个新文件，跑测试、抓网页、调 LSP 都行，第二篇文章末尾手把手带你写第一个。
- **改系统提示词。** `prompt.py` 才 33 行，改一句就能看到 agent 的脾气变了，是门槛最低的「改一处就有反馈」。
- **直接当库 import。** 顶层导出了 `Agent`、`LLM`、`Config`，能嵌进你自己的程序：

```python
from corecoder import Agent, LLM

llm = LLM(model="deepseek-chat", api_key="sk-...", base_url="https://api.deepseek.com")
print(Agent(llm=llm).chat("找出项目里所有 TODO 注释并列出来"))
```

往深里做，方向也都摆在明处。下面这些 CoreCoder 都没做，是设计取舍，不是没做完；换个角度，每一条都是你能接着往下做、把它推向更强的入口：

- **bash 的危险命令拦截只是正则黑名单。** 防手滑，不是安全沙箱。要面对不可信输入，就得上 seccomp 或容器隔离。这条最硬，要一路走到系统调用和隔离那一层。
- **重试只做了指数退避。** 没有 fallback 模型，也没有美元硬预算。顺着 `llm.py` 往下，加一条 fallback 模型链和超预算自动停的闸，改动基本就集中在这一个文件。
- **后台委派仍保持最小实现。** Worktree 隔离、分阶段 Agent Team、有界重试、熔断、基于游标的进度、进程租约、显式加密队列和租约分片的多工作区 Worker 池现在共用一个控制器；更丰富的可选遥测仍是自然的下一处扩展点。
- **不做 MCP，不做 RAG。** 接上 MCP 让它用上外部工具生态，或给大仓加检索式的代码定位，都是从「最小核心」往「你自己的更强 agent」扩的真实方向。

README 只给方向，每条的代码细节第七篇接着讲。挑一个动手，就是把它做得更好的开始。

## 命令

进了 REPL，`/help` 列全部，常用的这几个：

```
/model <名称>    切换模型
/compact         手动压缩上下文
/tokens          查看 token 用量和费用估算
/diff            查看本次会话改过的文件
/undo            撤销写入/编辑工具的修改（`/undo force` 强制覆盖冲突）
/save  /sessions 手动检查点 / 列出全部会话
/history [n]     查看全部历史 / 最近 n 轮
/memory          查看跨会话记忆和待反思会话
/memory show <id> / search <查询> / archive <id> / approve <id> / reflect
/skills          列出内置、用户和项目 Skill
/skill search <查询> / show <id> / use <id> / unuse <id> / explain / audit / metrics / evolve <memory-id>
/permissions [user|session|project|builtin] 查看规则及稳定 ID
/permissions clear-session /revoke <id> 删除可变授权
/security explain <工具> <JSON|bash 命令> 仅预演策略，不执行
/audit [筛选] [条数] [tool=<名称>] 查询最近的安全决策
quit / exit      退出（Ctrl+C 取消当前回合）
```

完整对话会在每轮结束后自动保存到 `~/.corecoder/sessions`；`/save` 可手动建立检查点，`/sessions` 列出全部历史会话，`corecoder -r <ID>` 可恢复续聊，并在交互模式下回显已保存的用户与助手消息。`/history` 可再次查看全部历史，`/history <n>` 只显示最近 n 轮。版本 2 会话文件使用独立展示记录，因此模型上下文压缩不会再删除原始对话；模型/工具消息单独保存，工具结果默认隐藏。旧版会话可兼容读取，并在下次保存时自动升级。会话 ID 会先清洗成安全字符，恶意名称无法路径穿越。

`/undo` 会恢复当前 CoreCoder 进程中由 `write_file`、`edit_file` 或 `edit_ast` 修改文件的原始字节，并删除这些工具新建的文件；同一文件即使被多次编辑，也会回到第一次修改之前。如果文件在 Agent 最后一次写入后又被外部修改，普通撤销会将其保留并报告冲突；只有显式执行 `/undo force` 才会覆盖。撤销历史在 `/reset` 后仍保留，但不会跨进程或随恢复会话持久化。`bash` 产生的任意文件系统副作用无法可靠追踪，不在撤销保证范围内。

### 多层安全审查

启用 `Guard` 后，每次 Tool 调用依次经过：不可覆盖的内置硬边界、Tool 的读写范围/网络能力/副作用/基线风险声明自检、确定性风险下限、可选语义/AI 风险复核、人工确认、频率限制和 JSONL 审计。AI 复核只能提升级别，不能降低确定性规则的结论。确认界面会展示命令或目标、能力范围、副作用、风险等级、网络目标以及上传、重定向和凭证标记；高风险发布、远端写入、基础设施变更和不可逆恢复只能单次确认，非交互环境默认拒绝。`a/always` 只对普通权限提示开放，产生的 allow 规则仅存在于当前进程；显式 `/permit` 才会持久化普通策略。规则具有稳定的 `usr-...`、`ses-...`、`prj-...` 和 `sys-...` ID；`/revoke` 只能删除用户或会话规则，项目规则与内置硬边界不可从 CLI 删除。每次 CLI 权限变更都会进入审计。`/security explain` 可以在不执行、不弹确认、不写入决策审计且不消耗频率配额的情况下预览最终规则、能力、风险和网络判断。

所有可读取外部文本的 Tool 输出都会携带 `[UNTRUSTED_TOOL_OUTPUT ...]` 来源标记。Guard 会先脱敏，再检测指令覆盖、伪造角色、凭证诱导和嵌入式工具调用；命中时添加 `[SECURITY_FINDINGS]`，但把原内容继续作为证据而不是指令。同一回合后续有副作用的调用会被污染传播机制提升为必须人工确认，来源和风险标记也会在上下文压缩后保留。审计日志位于 `~/.corecoder/audit`，记录规则来源、能力范围、风险等级、确认结果、不含任意载荷原值的参数摘要和参数摘要哈希。使用 `/audit allow|deny|flag|policy|confirmed [1-100] [tool=<名称>]` 可以筛选当天记录；损坏行会被报告并跳过，不会遮蔽其余有效审计历史。

文件写入会拒绝系统目录、凭据目录、真实 `.env` 和私钥目标，包括尚不存在的凭据目录。网络出口默认采用 `confirm` 模式：云元数据和链路本地地址硬拒绝，私网、未知域名、重定向、携带认证信息及远端写入要求确认。可用 `CORECODER_NETWORK_MODE=deny` 拒绝非 allowlist 目标，并通过逗号分隔的 `CORECODER_NETWORK_ALLOWLIST=api.example.com,*.pythonhosted.org` 配置精确域名或子域通配符；私网和元数据地址不能加入 allowlist。

设置 `CORECODER_SANDBOX=1` 后，如果 Docker 不可用则拒绝 Bash 调用，不会静默降级到宿主机执行。容器网络默认为 `CORECODER_SANDBOX_NETWORK=none`；只有显式设置为 `bridge` 才开放容器网络。命令预检无法防御 DNS rebinding 或任意代码在运行时自行联网，生产环境仍应使用容器网络策略、出口代理或主机防火墙实施强制控制。

集成真实 AI 分类器时，向 `Guard(risk_reviewer=...)` 传入同步回调，返回 `RiskAssessment`。分类器超时或异常时，有副作用的调用会提升为高风险并要求确认；不要让分类器直接执行工具或修改权限。

每个完整回合结束后，CoreCoder 会先快速写入持久化 pending 检查点，再由后台 worker 提取稳定的用户偏好、用户画像、项目约定、历史反馈、经过验证的程序性经验和有价值的任务情景，并保存到 `~/.corecoder/memory`。尚未处理的连续回合会合并，基于 checkpoint token 的确认机制保证旧提取任务不会误删更新的聊天。启动恢复也在后台运行，正常退出不再等待模型请求；未完成的工作会留在 pending，供后续运行继续处理。此后每轮用户输入都会按当前问题和项目范围重新检索活跃记忆，使用次数与成功/失败反馈会参与后续排序；检索也会搜索用户原始 evidence，因此中文请求被总结成英文后仍能用中文召回。`/memory` 会显示 pending 的重试次数和最近提取错误，连续失败三次后进入隔离区。记忆支持证据、版本、候选/活跃/归档/替换状态、自动重建的 `MEMORY.md` 以及跨进程更新锁。

使用 `/memory show <id>` 查看详情、`/memory search <查询>` 搜索、`/memory archive <id>` 归档、`/memory approve <id>` 显式启用、`/memory reflect` 重试待处理会话、`/memory forget <id>` 永久删除。设置 `CORECODER_MEMORY=0` 可关闭，`CORECODER_MEMORY_DIR` 可修改目录。程序性记忆必须有真实成功工具调用和原文验证证据；若常规提取遗漏 procedure 或返回错误格式，独立的受约束提炼仍可保存已验证的执行记忆，pending 恢复会优先运行这一结构化通道。情景记忆必须有真实失败工具调用以及有证据的失败或根因经验，两者都限制为项目作用域。“运行、分析、总结任务”的请求即使提到以后复用，也不会被当作用户画像或项目记忆；经过验证的复用步骤应保存为 procedure。历史上仅由此类任务证据生成的误判文件仍保留用于审计，但不再参与检索。新生成的执行类记忆首先进入不可检索的 candidate 状态，第二个独立会话再次验证后自动晋升 active，也可以用 `/memory approve <id>` 人工启用。反思前会折叠重复 Replay 噪声。修改持久化 `.corecoder/permissions.json` 必须得到用户明确确认，代理不能为了绕过命令拦截而静默改写。系统不会根据记忆自动生成或安装可执行 Skill，Skill 晋升仍需单独审核。

多用户部署可设置 `CORECODER_TENANT_ID` 和 `CORECODER_USER_ID`。启用后，记忆、用户 Skill 和 Skill 效果统计会写入独立的 `tenants/<tenant>/users/<user>` 命名空间；不设置时继续使用原有单用户目录，保持向后兼容。标识符会经过严格验证，不能包含路径分隔符或目录穿越片段。记忆文件读取带有变更感知缓存，检索使用倒排候选集，避免每次请求重复解析和逐条计算全部记忆。

### Skill

Skill 是构建在原子 Tool 之上的可复用任务指导，分为 `atomic`、`workflow` 和 `orchestrator` 三层。CoreCoder 会发现包内置 Skill、`~/.corecoder/skills` 下的用户 Skill，以及项目 `.corecoder/skills` 下的项目 Skill；同一 ID 按项目、用户、内置的顺序覆盖。每个 Skill 包含一份轻量 `skill.json` Catalog 元信息和完整 `SKILL.md` 指令。

路由采用“2+1”渐进加载：轻量内存 Catalog 合并精确、标签/签名、上下文、对比例和可选语义召回，只保留少量候选；随后结合正例、硬负例、Tool/输入/上下文/权限前置条件、依赖/冲突、历史失败惩罚、作用域和 Prompt 成本精排；最后只加载选中 Skill 的 `SKILL.md`。清单可用 `resource_modes` 声明模式级资源，使系统只读取本次命中的 reference，并只暴露对应 script/asset 路径。默认仅自动选择一个主 Skill和最多两个辅助 Skill，辅助 Skill 必须通过 `dependencies` 或 `composes_with` 明确建立关系。

路由结果分为 `explicit`、`auto`、`clarify` 和 `abstain`。v2 高置信度且差值充分时自动启用；中置信度或候选接近时直接返回一个澄清问题，不调用 LLM、也不暴露 Tool；匹配过弱时不强行加载 Skill。请求中的 `$skill.id` 或 `/skill use <id>` 可以显式启用，`不要使用 $skill.id` 可只在当前回合排除它。通过 `/skill explain` 查看召回分、精排分、置信度、任务签名和拒绝原因，通过 `/skill audit` 查看缺失、循环、矛盾、替代和高重叠关系。Skill 只能收紧 Tool 范围；高风险 Skill 在首次有副作用的 Tool 前强制确认，并继续经过原有安全检查。

`skill.json` v2 可渐进增加以下字段；v1 清单保持兼容：

```json
{
  "schema_version": 2,
  "layer": "workflow",
  "signature": {
    "domains": ["testing"],
    "actions": ["debug", "修复"],
    "objects": ["pytest", "测试失败"],
    "artifacts": ["Python test"],
    "outputs": ["passing tests"],
    "constraints": []
  },
  "examples": {
    "positive": ["pytest is failing"],
    "negative": [],
    "hard_negative": ["add new unit tests"],
    "contrastive": [
      {"query": "add tests", "expected_skill": "testing.test-generation"}
    ]
  },
  "relations": {
    "dependencies": [],
    "composes_with": ["testing.test-generation"],
    "supersedes": []
  },
  "routing": {"allow_implicit": true, "risk": "medium", "rollout_percent": 100}
}
```

生命周期支持 `draft → candidate → shadow → canary → active → deprecated`。`shadow` Skill 只参与打分观察，不会被激活；`canary` 使用稳定路由键按 `rollout_percent` 放量；`supersedes` 会把旧能力的隐式匹配重定向到继任者；`SkillManager.transition` 对可编辑 Skill 的发布或回滚执行状态校验并写入审计记录。宿主还可传入附件/载体、必要输入、连接应用、live app、权限、外部写入、风险、意图类型和稳定放量键。`corecoder.skills.evaluate_router` 可计算正例 Precision@1、整体准确率、误激活率、漏召回率、澄清率、用户改选率、任务成功率、置信度差值、P95 路由延迟、高风险确认率、Shadow 对比、候选数量及估算加载 Token。

设置 `CORECODER_SKILLS=0` 可关闭路由；`CORECODER_SKILLS_DIR`、`CORECODER_SKILL_TOP_K`、`CORECODER_SKILL_MAX_ACTIVE` 和 `CORECODER_SKILL_PROMPT_CHARS` 分别控制用户目录和路由预算。`CORECODER_SKILL_MIN_SCORE`（默认 `0.24`）是候选下限；`CORECODER_SKILL_CLARIFY_CONFIDENCE`（默认 `0.65`）、`CORECODER_SKILL_AUTO_CONFIDENCE`（默认 `0.82`）和 `CORECODER_SKILL_AMBIGUITY_MARGIN`（默认 `0.12`）共同控制拒绝、澄清和自动调用。

每次 Skill 路由及其终态结果都会以聚合计数写入用户命名空间下的 `.telemetry.json`，不额外保存提示词或 Tool 输出。积累足够样本后，失败和部分成功会生成有上限的历史惩罚并立即参与后续精排；少量样本不会触发调权。使用 `/skill metrics` 查看路由次数、成功/部分成功/失败以及当前惩罚值。

大规模部署可通过 `semantic_recaller(query, limit)` 接入向量数据库或 ANN 服务，让外部索引直接返回 Skill ID 和相似度；兼容的逐 Skill `semantic_scorer` 仍可用于小目录，但批量召回接口不会扫描整个 Catalog。

`/skill evolve <memory-id>` 可以把已经激活且经过至少两个独立会话验证的项目 procedure 记忆生成到项目 `.corecoder/skills`。生成结果固定为 `candidate`，关闭隐式调用、Canary 放量为零并带来源记忆；生成过程绝不会直接启用。维护者审查步骤、Tool、权限、边界、示例和验证标准后，才能显式推进 `shadow → canary → active` 生命周期。

程序记忆验证现在同时要求最终交付物完成、执行边界满足和可核对的验证证据。执行异常、未完成报告、交接摘要、访问凭据文件都不能贡献验证次数。待处理检查点保留压缩前的本轮执行证据。`/memory show` 会显示 `Completion-checked sessions`；旧版 procedure 的计数不能直接用于召回或 Skill 进化，`/memory approve` 也不会补造验证证据，需经过两个按新规则完成的独立会话重新验证。`/skill metrics` 增加澄清次数。

`read_file` 和 `grep` 在读取前拦截真实 `.env`、凭据目录和私钥文件；递归 grep 自动跳过这些文件。`.env.example`、`.env.sample` 和 `.env.template` 仍可读取。这是内容读取工具的边界，不是通用文件系统沙箱；脱敏替换标记不能作为原文件含有占位符的证据。

DeepSeek 请求保留服务端返回的思考字段，并为合成的助手消息补齐空字段。上下文压缩保留当前请求；空白、截断或仅含交接摘要的回答会获得一次有界的最终输出重试，失败仍记为未完成。离线回归执行 `python -m pytest -q`；真实接口测试需显式设置 `CORECODER_LIVE_TESTS=1` 后执行 `python -m pytest -q tests/test_provider_live.py`，使用已配置的 DeepSeek 接口，仅发送虚构数据。

## 相关项目

如果你读 CoreCoder 读得还顺，下面几个我做的 agent / LLM 系统方向的工具也许用得上：

- **[RepoWiki](https://github.com/he-yufeng/RepoWiki)** — 被丢进一个陌生代码库？它给你一份带「从哪读起」路径的 wiki，一个可自托管的 DeepWiki 替代。
- **[FindJobs-Agent](https://github.com/he-yufeng/FindJobs-Agent)** — 别再手动刷招聘网站：它按你的简历给岗位排序，还能跑模拟面试。
- **[ContractGuard](https://github.com/he-yufeng/ContractGuard)** — 签字前先把有风险的条款挑出来：它读合同、标出危险点。
- **[GitSense](https://github.com/he-yufeng/GitSense)** — 想给开源做贡献？它帮你找到值得做的 issue，还能估你的 PR 多大概率被合。
- **[CodeABC](https://github.com/he-yufeng/CodeABC)** — 不会写代码也能看懂一个项目，专给小白做的。

## 贡献 / License

动手之前先跑一遍 `pytest tests/ -q`（86 个测试）、`ruff check` 和 `compileall`，绿了再提。MIT License，欢迎 fork 拿去造更好的东西，能在 README 里留一句出处就更好。

---

作者 [何宇峰](https://github.com/he-yufeng)，曾任职 Moonshot AI (Kimi)。早前写过一篇相当完整的 [Claude Code 源码分析](https://zhuanlan.zhihu.com/p/1898797658343862272)，这个项目是它的动手版：那篇带你读懂，这个带你重建。

> CoreCoder 原名 NanoCoder，为避免和 [Nano-Collective/nanocoder](https://github.com/Nano-Collective/nanocoder) 混淆而改名，旧链接会自动跳到这里。
