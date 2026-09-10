<div align="center">

# CoreCoder

**The nanoGPT of coding agents. 1,081 lines of pure Python — understand how a coding agent actually works, then fork your own.**

*learn from it · fork it · ship something better*

[中文](README_CN.md) | English | [Source-reading series · 8 bilingual essays](article/00-index_EN.md)

[![PyPI](https://img.shields.io/pypi/v/corecoder)](https://pypi.org/project/corecoder/)
[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://github.com/he-yufeng/CoreCoder/actions/workflows/ci.yml/badge.svg)](https://github.com/he-yufeng/CoreCoder/actions)
[![engine](https://img.shields.io/badge/engine-1081_LoC-blue)](article/00-index_EN.md)
[![essays](https://img.shields.io/badge/source--reading-8_bilingual-orange)](article/00-index_EN.md)

</div>

- **Readable end to end.** Read the whole engine in an afternoon, with no magic hidden anywhere you can't follow it.
- **Hackable.** Set a breakpoint on any line, change it, rerun, all on your own machine. It genuinely works, which makes this a living reference rather than a diagram.
- **The gaps are the point.** It deliberately keeps only the minimal core; what's missing isn't half-finished, it's where you branch off and make it your own.

## How it compares

| | CoreCoder | Claude Code | aider | nanoGPT |
|---|---|---|---|---|
| Lines of code | ~1,081 engine / 1,714 total | hundreds of thousands (closed) | tens of thousands of Python | ~600 (two files) |
| Time to read it all | one afternoon | can't (closed) | a few days of slogging | one afternoon |
| Breakpoint, change, rerun? | yes, every line | no | yes, but there's a lot | yes |
| What it's for | understand one, then fork your own | production coding assistant | terminal pair-programming | minimal GPT for teaching |

The nanoGPT column is there as a reference point: minimal, readable, but it teaches you to train a GPT. CoreCoder is after the same thing, only the subject is an agent that actually edits code. Sitting it next to Claude Code and aider isn't about competing for their users. CoreCoder is the foundation you stand on while you learn from them and get going; it isn't in the same race.

## What this is

I've always felt coding agents get talked about as if they were arcane. Strip a tool like Claude Code or Cursor all the way down and the core is a `while` loop wrapped around a large model, plus seven or eight tools that let it actually do things. The hard part was never the loop; it's everything the loop has to cope with once it meets the real world. CoreCoder is the minimal version that writes that core out honestly.

The engine (loop, model interface, context, tools, sessions) is 1,081 lines once you drop blank lines and comments. Counting the outer CLI, config and packaging too, the whole package is 18 files: 1,714 physical lines, 1,385 net, every one short enough to read in a single sitting.

And it really runs: reads and writes files, executes shell, spawns sub-agents, compacts context in three tiers, and tells you the tokens and dollars a run burned whenever you ask. 86 tests, all green. But the point of it running isn't to become your daily driver. It runs so the walkthrough can't lie: a reference that shows how an agent works has to actually work.

The code came out of a public teardown: open analyses have already exposed a lot of the load-bearing architecture inside production agents like Claude Code. I took the most essential layer and rewrote it honestly, in as little code as I could. So reading CoreCoder is roughly like reading a runnable, annotated take on how that kind of agent works, except it's only a minimal reimplementation, sitting right there on your machine for you to take apart and change.

<p align="center">
  <img src="https://raw.githubusercontent.com/he-yufeng/CoreCoder/main/assets/demo_en.png" width="760"
       alt="A real CoreCoder run: corecoder -p asks it to fix buggy.py; the agent reads the file, edits the code, runs it to confirm, and reports what it changed.">
</p>

<p align="center"><sub><i>These thousand lines really do run a full loop end to end: ask it to fix buggy.py and it reads the file, edits the code, runs it once to confirm, then reports back on its own. Watch it, then come back and read the code.</i></sub></p>

This README follows the same arc: the first half helps you **read it** (the code map, the main loop, eight essays), the second half helps you **fork it** and points at a few directions worth pushing further.

## Run it once first (five minutes before you read)

Before you read the source, get it running on your machine once to build some intuition. It's a foundation meant for forking, so the recommended path is to clone it and install editable, reading and changing as you go:

```bash
git clone https://github.com/he-yufeng/CoreCoder
cd CoreCoder
pip install -e .
```

If you just want to get it running first, `pip install corecoder` works too.

Give it a model and a key and it goes. It speaks the OpenAI-compatible API by default, and switching providers is usually just two environment variables:

| Provider | Example env vars |
|---|---|
| OpenAI (default `gpt-5.5`) | `OPENAI_API_KEY=sk-...` |
| DeepSeek | `OPENAI_API_KEY=sk-... OPENAI_BASE_URL=https://api.deepseek.com CORECODER_MODEL=deepseek-chat` |
| Local Ollama | `OPENAI_API_KEY=ollama OPENAI_BASE_URL=http://localhost:11434/v1 CORECODER_MODEL=qwen2.5-coder` |

Kimi, Qwen and the like are the same two variables; for providers that don't even offer an OpenAI-compatible endpoint, the optional LiteLLM backend (`pip install "corecoder[litellm]"`) routes to a hundred-plus of them. The third essay goes into this in detail. The key can be `export`ed directly or dropped into a `.env` at the project root, which is loaded on startup. Then:

```bash
corecoder                                             # interactive REPL
corecoder -p "add error handling to parse_config()"   # one-shot mode, exits when done
```

## Read it: the code map

Laid out flat, the whole project is this big. Skim it before you clone and you'll know where everything is. This is the most concrete difference from Claude Code's hundreds of thousands of lines: you can read it like the table of contents of a book. Start from the main loop in `agent.py`; that's the heart of the whole agent.

```
corecoder/
├── agent.py        agent loop + parallel tool exec       150 lines   ← start here
├── llm.py          streaming client + retry + cost        336 lines
├── context.py      three-tier context compaction          210 lines
├── session.py      save / resume + path-traversal guard    97 lines
├── prompt.py       system prompt                           33 lines
├── cli.py          REPL + slash commands + one-shot       270 lines
├── config.py       env-var config                          57 lines
└── tools/
    ├── bash.py       shell + dangerous-command gate + cd  127 lines
    ├── edit.py       unique-match search/replace + diff    92 lines
    ├── grep.py       content search                        79 lines
    ├── glob_tool.py  filename matching                     47 lines
    ├── read.py       file read                             53 lines
    ├── write.py      file write                            38 lines
    ├── agent.py      sub-agent spawning                    58 lines
    └── base.py       tool base class                       27 lines
```

Seven tools: `bash`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, and `agent` (which spawns a sub-agent). Everything else is the CLI shell, config, and packaging wrapped around that engine core.

## A `while` loop is the whole agent

The whole of an agent fits in one sentence: hand the user's words to the model, run whatever tools it asks for, stuff the results back into the context, ask again, and keep going until it stops asking for tools and gives an answer. In code, that's about a dozen lines:

```python
# corecoder/agent.py · the main loop (trimmed skeleton)
def chat(self, user_input):
    self.messages.append(user_input)

    for _ in range(self.max_rounds):                   # bounded, so it can't run away
        reply = self.llm.chat(self.messages, self.tools)   # ask the model what to do next
        if not reply.tool_calls:                       # model wants no more tools
            return reply.text                          #   -> done, hand the answer back
        results = run_parallel(reply.tool_calls)       # tools requested -> run in parallel
        self.messages += results                       # feed results back, loop again

    return "(hit the round limit)"
```

That's the whole thing. The core skeleton is about twenty lines; counting parallel execution and the bookkeeping after a Ctrl+C interrupt, maybe forty. Almost everything else in CoreCoder's thousand-odd lines is there to clean up the mess the loop runs into once it meets the real world. `llm.py` ends up the biggest file in the project, not because calling a model is hard, but because a streamed response splinters each tool call's arguments into fragments you have to restitch in order, a provider will hand you half a JSON object or a null `usage` field, and 429s, timeouts, dropped connections and 5xx all need backoff-and-retry while the other 4xx should just raise. That unglamorous grunt work, not the loop, is where the real engineering of taking an agent from demo to delivery actually lives; the third essay follows it down to the line.

Three decisions are worth a closer look, because they're the kind of call you can only make after you've understood how others did it, and they're judgments you can lift straight into your own fork.

**`edit_file` does search-and-replace on a unique match, not line numbers.** Line numbers are a trap: the model only has to miscount by one and it quietly edits the wrong place. Anchor on a unique snippet of the original instead. If there's no match, it hands the start of the file back so the model can re-anchor; if there are several matches, it makes the model bring more surrounding context rather than gamble on one. On a successful edit it returns a diff. Recoverable on failure, verifiable on success: the whole loop stays inside the tool.

**Context isn't cut all at once when it's full; it gives ground in three tiers, cheapest first.** At half full (50%) it trims over-long tool outputs in place, a tier that's purely mechanical and costs no model call. If 70% still isn't enough, it has the model summarize the older turns into a single paragraph while keeping the most recent ones verbatim. Only at 90% does it hit the emergency tier and pull everything, summary and recent turns alike, down to its tightest form. Blunt truncation tends to throw away exactly the early decision a long task leans on most; tiering lets it surrender the least important things first instead of lopping off the oldest decisions wholesale from the start.

Large tool observations are externalized before they first enter history. The transcript keeps a deterministic preview and an `artifact://sha256/...` reference, and the read-only `retrieve_context` tool can recover exact keyword matches or line ranges later. At the summary watermark, older turns become a schema-validated JSON checkpoint containing the current goal, constraints, decisions, files, verification, errors, pending work, and artifact references. Checkpoints are versioned and only rewritten after a low-water rearm or a meaningful batch of new messages, keeping prompt prefixes stable between compactions. Set `CORECODER_CONTEXT_ARTIFACTS=0` to disable artifact storage; `CORECODER_CONTEXT_ARTIFACTS_DIR` and `CORECODER_CONTEXT_ARTIFACT_THRESHOLD` control storage and the default 12,000-character cutoff. `CORECODER_CONTEXT_ARTIFACT_TTL_DAYS` (default 30) and `CORECODER_CONTEXT_ARTIFACT_MAX_MB` (default 256) bound retention; expired artifacts are removed before the oldest entries are evicted for capacity. `/tokens` reports provider cache hit/miss usage when available, plus externalization, retrieval, pruning, compaction, and checkpoint counters.

**Sub-agents now run behind one parent-owned control plane.** Every delegation is a validated `TaskSpec`: objective, minimal context, exact tool allowlist, read/write roots, token and tool-call budgets, timeout, role, workspace mode, and acceptance criteria. `TaskController` owns the state machine, concurrency cap, timeout and cancellation. A child has an independent history, cannot receive the `agent` tool, cannot ask for permission elevation, and returns a bounded `TaskResult`; runtime code—not the child model—derives status, changed files, usage and policy violations. The parent still performs final review and acceptance.

Library callers can use the protocol directly:

```python
from corecoder import Agent, TaskRole, TaskSpec

spec = TaskSpec(
    objective="Inspect authentication and identify its entry points",
    role=TaskRole.RESEARCHER,
    allowed_tools=("read_file", "grep", "glob"),
    read_paths=("corecoder/security",),
    token_budget=8_000,
    timeout_seconds=120,
    acceptance_criteria=("Cite the defining files",),
)
result = await agent.delegate(spec)
assert result.requires_parent_review
# After independently checking each criterion:
# accepted = agent.accept_task(result.task_id, parent_verified_checks)
```

CoreCoder uses LangGraph as its single user-facing orchestration path. The
native adapter remains only as an internal compatibility and evaluation baseline:

```bash
pip install -e .
corecoder
```

```python
from corecoder import Agent, TaskSpec, WorkflowRequest

agent = Agent(llm=llm)
workflow = await agent.run_workflow(WorkflowRequest(
    task=TaskSpec(objective="Implement and verify the scoped change"),
    max_replans=1,
    max_total_tokens=24_000,
))
```

The graph is `plan -> approval -> execute -> verify -> review -> decide`. It does
not receive tools or filesystem access: execution still goes through
`TaskController`, and a planner may refine instructions but cannot broaden the
task's role, tools, paths, budgets, timeout, acceptance criteria, or execution
mode. Policy violations and exhausted budgets are never retried; write retries
require worktree isolation. Unscoped high-risk tools interrupt before execution
and resume only with a decision bound to the exact task digest.

There is no native execution bypass in the public Agent API. Every main-agent
and child-agent `chat()` turn runs through a LangGraph turn lifecycle
(`plan -> execute -> verify -> review`); foreground delegation, background and
durable work, batch delegation, and team stages additionally run through the
scoped task graph above. The raw model/tool loop and direct controller executor
are private graph-node callbacks, which prevents orchestration recursion while
keeping one user-visible execution path. `agent.last_turn_workflow` exposes the
most recent turn trace for diagnostics.

In-memory checkpoints are the default. Library callers that need restart
recovery can pass `encrypted_sqlite_checkpointer(path, key=...)`; only strictly
allowlisted protocol and control-plane types are serialized, and the AES key
must be 16, 24, or 32 bytes. The built-in planner/verifier/reviewer are deliberately deterministic:
LangGraph improves control flow, not model intelligence. Inject independent
callbacks when stronger planning or review is needed, and keep parent-side test
and acceptance verification as the authority.

`measure_workflow()` and `summarize_workflows()` derive comparable success,
retry, token, latency, tool-call, policy-violation, reported-test, and
parent-verified-acceptance metrics from structured results. This makes Native
versus LangGraph A/B evaluation possible without trusting model-written prose.

`agent.tasks.snapshot(task_id)`, `list_tasks()` and `events()` expose bounded,
prompt-free control-plane state for monitoring. Lifecycle events are also written
to the existing JSONL audit log with task, parent, child, role, permission scope,
and workspace identity. Audit failures never change task execution, and retained
in-memory task history is bounded by `task_history_limit`.

Long-running tasks can be submitted without blocking the caller:

```python
task_id = await agent.submit_task(spec)
snapshot = agent.tasks.snapshot(task_id)
result = await agent.wait_task(task_id, timeout=30)  # waiter-only timeout
# agent.cancel_task(task_id)                         # explicit task cancellation

# Cursor-based progress: tool names and controller milestones, never arguments.
batch = await agent.wait_task_events(task_id, after_sequence=0, timeout=30)
cursor = batch.next_sequence
```

Cancelling or timing out a waiter does not cancel the background task. Only an
explicit control-plane cancellation (or the task's own deadline) stops execution.
The main model can request the same behavior with `agent(background=true)` and
then use the parent-only `task_control` tool. In the interactive CLI, `/tasks`,
`/task <id>`, `/wait-task <id> [seconds]`, and `/cancel-task <id>` expose the
same operations; `/watch-task <id> [seconds]` streams cursor-based progress.
The CLI keeps its event loop alive while waiting for terminal input, so
submitted work continues between user commands. Progress events contain only
controller milestones and tool names—not tool arguments, model tokens, or
chain-of-thought. Clients receive a `history_truncated` flag if their cursor has
fallen behind bounded retention.

The CLI also keeps a bounded, append-only task journal under
`~/.corecoder/tasks`, scoped by tenant/user and keyed by a digest of the
workspace path. A later process in the same workspace restores terminal results
for `/tasks`, `/task`, and `/wait-task`. Work that had no durable terminal result
is marked `interrupted` and is never automatically replayed. The original
`TaskSpec.objective` and context are not stored; terminal `TaskResult` content is
stored so the parent can inspect the outcome. Set `CORECODER_TASK_PERSISTENCE=0`
to disable this or `CORECODER_TASK_STATE_DIR` to move the journal.

A heartbeat lease gives exactly one process scheduling and journal-write
ownership for each workspace. A second process opens the same durable history
in read-only observer mode: it can refresh, wait, and watch, but cannot submit or
cancel delegated work. `/claim-tasks` explicitly acquires a released or stale
lease; only then are genuinely abandoned tasks marked `interrupted`. A live PID
on the same host is never displaced solely because its heartbeat is old. For
shared storage across hosts, `CORECODER_TASK_LEASE_STALE_SECONDS` controls the
stale-heartbeat threshold (default 30 seconds, minimum 5).

Durable execution is explicit. Set `durable=True` only on a background
`TaskSpec` (or use `agent(background=true, durable=true)`) to place its complete
specification in an authenticated Fernet-encrypted queue before scheduling. The
CLI resumes valid queued work after it acquires the workspace lease. Normal
terminal results and explicit cancellation remove the item; a crash or shutdown
keeps unfinished work for the next owner. Tampered, wrongly keyed, malformed, or
over-capacity items are never executed. The generated queue key is stored as
`.task-queue.key` in the tenant/user task-state directory with restrictive file
permissions where supported. Durable mode through the model-facing tool requires
a fresh user confirmation because objective and context are persisted. Ordinary
tasks retain the prompt-free journal behavior.

Run a dedicated foreground consumer from the repository it should own:

```bash
corecoder worker                       # poll continuously (default: every 1s)
corecoder worker --poll-interval 0.25 # custom bounded polling interval
corecoder worker --once               # drain the current queue and exit
corecoder worker --workspace ../repo-a --workspace ../repo-b
corecoder worker --workspace ../repo-a --workspace ../repo-b --workspace-concurrency 2
```

The worker uses the same `Agent`, lease, controller, tool boundaries, budgets,
worktree merge, audit, and encrypted queue as the interactive CLI. It runs with
no confirmation callback, so operations that require new human approval fail
closed; durable submission must already have been approved by the producing
client. A non-owner or persistence-disabled worker exits non-zero. Ctrl+C marks
shutdown before asyncio cancels children, preserving unfinished durable items
for the next worker. While a worker owns the lease, another CLI may enqueue an
explicit durable task but still cannot directly execute or cancel delegated
work. Repeating `--workspace` starts a pool with a separate Agent, tool registry,
controller, queue and lease for each resolved directory. Queue scans run
concurrently, so a busy repository cannot delay admission in another. Leases
provide cross-process sharding: workspaces already owned by another live worker
are skipped, while available workspaces continue normally; duplicate or missing
workspace paths are rejected before execution.
The pool-wide execution cap defaults to four and can be set from 1 to 32 with
`--workspace-concurrency`; per-workspace controller limits still apply beneath it.
In `--once` mode, both worker forms report succeeded and failed task counts;
any terminal task failure or worker error produces a non-zero exit status.

Both execution backends use the same protocol. `fork` operates in the shared tree and defaults to one child at a time. `worktree` requires a clean Git parent, runs the child in a detached managed worktree, collects a binary diff, checks it centrally with `git apply --check`, applies it to the parent, and records the result in `/undo`; a conflict leaves the parent untouched and retains the isolated tree for inspection. `bash` and `undo_changes` have no reliably enforceable path argument and are rejected in delegated specs unless a library caller explicitly opts into unscoped tools; worktree tasks reject them unconditionally.

`Agent.run_team()` and `agent(mode="coding_team")` provide a staged researcher → executor → reviewer template over that controller. Only the parent forwards bounded summaries between stages. Read-only failures may opt into up to three budget-sharing retries, and repeated failures open a controller circuit breaker before more child work is admitted.

Every one of these *whys* is traced down to the actual lines of code in the series below.

## The source-reading series · 8 bilingual essays

I also wrote a bilingual source-reading series, one intro plus seven parts, each in Chinese with an English mirror. Against CoreCoder's actual code, it walks through how agents like Claude Code work under the hood. One hard rule I set myself: every line count and every snippet is re-read and re-checked from the repo, never written from memory. The first six get you reading, the seventh gets you forking; read them in any order.

- **[Intro · Read Claude Code through CoreCoder, then build your own](article/00-index_EN.md)**
- **[01 · An agent, at its core, is a `while` loop](article/01-the-loop_EN.md)** — the main loop in `agent.py`, interrupts, and the round limit
- **[02 · The tool system: letting the model act, safely](article/02-tools_EN.md)** — the seven tools in `tools/` and the bash safety gate
- **[03 · Plug in any LLM, and keep the bill honest](article/03-llm-and-cost_EN.md)** — `llm.py`'s provider wrapper, retries, and cost accounting
- **[04 · Surviving a long task on a finite window](article/04-context_EN.md)** — `context.py`'s three-tier compaction and orphaned tool messages
- **[05 · Parallel execution and sub-agents](article/05-parallel-and-subagents_EN.md)** — thread-pool concurrency and sub-agent isolation
- **[06 · Turning it into a real command-line tool](article/06-session-and-cli_EN.md)** — `session.py` and path-traversal defense
- **[07 · Fork CoreCoder into your own coding agent](article/07-build-your-own_EN.md)** — from fork to custom tools to swapping models

## Fork it, build something better

Once you understand it, the natural next step is to fork. Getting started doesn't take much:

- **Swap in a model you actually use.** It's the two env vars from above; `llm.py` (336 lines) is the entry point for all provider adaptation.
- **Add a tool of your own.** Write a new file against the tool base class in `tools/base.py` (27 lines): run tests, fetch a page, call an LSP, whatever. The end of the second essay walks you through your first one by hand.
- **Rewrite the system prompt.** `prompt.py` is all of 33 lines; change one line and you'll watch the agent's temperament shift. It's the cheapest "change one thing, see a result" in the whole project.
- **Import it as a library.** The top level exports `Agent`, `LLM`, and `Config`, ready to embed in your own program:

```python
from corecoder import Agent, LLM

llm = LLM(model="deepseek-chat", api_key="sk-...", base_url="https://api.deepseek.com")
print(Agent(llm=llm).chat("find every TODO comment in this project and list them"))
```

Going deeper, the directions are out in the open too. None of the following is in CoreCoder, by design, not because it's unfinished. Flip it around and each one is an entry point you can carry into a real tool of your own:

- **The dangerous-command blocking in bash is just a regex blacklist.** It guards against slips, not a security sandbox. Facing untrusted input means reaching for seccomp or container isolation. This is the hardest of the four; it goes all the way down to the syscall and isolation layer.
- **Retry is only exponential backoff.** No fallback model, no hard dollar budget. Follow `llm.py` down and add a fallback model chain plus a stop-on-over-budget gate; the change stays mostly inside that one file.
- **Background delegation remains deliberately small.** Worktree isolation, staged Agent Teams, bounded retries, circuit breaking, cursor-based progress, process leases, an opt-in encrypted queue, and lease-sharded multi-workspace worker pools now share one controller; richer opt-in telemetry remains a natural extension point.
- **No MCP, no RAG.** Wire up MCP to give it the external tool ecosystem, or add retrieval-based code location for big repos. Both are real ways to grow from a minimal core into your own stronger agent.

The README only points; the seventh essay picks up the code details for each. Pick one and start; that's the whole reason the core is kept this small.

## Commands

Inside the REPL, `/help` lists everything; these are the ones you'll reach for:

```
/model <name>    switch model
/compact         compact the context by hand
/tokens          token usage and cost estimate
/diff            files changed this session
/undo            undo tracked write/edit changes (`/undo force` overrides conflicts)
/save  /sessions checkpoint / list all sessions
/history [n]     show all history / latest n turns
/memory          inspect memories and pending reflections
/memory show <id> / search <query> / archive <id> / approve <id> / reflect
/skills          list built-in, user, and project skills
/skill search <query> / show <id> / use <id> / unuse <id> / explain / audit
/permissions [user|session|project|builtin] show rules and stable IDs
/permissions clear-session /revoke <id> remove mutable approvals
/security explain <tool> <JSON|bash command> preview policy without execution
/audit [filter] [n] [tool=<name>] inspect recent security decisions
quit / exit      exit (Ctrl+C cancels the current round)
```

Complete conversations are automatically checkpointed after every turn under `~/.corecoder/sessions`; `/save` creates an explicit checkpoint, `/sessions` lists every saved conversation, and `corecoder -r <id>` resumes one and displays its saved user/assistant history in interactive mode. `/history` displays it again, while `/history <n>` limits output to the latest *n* user turns. Version-2 session files keep an independent display transcript, so context compression cannot erase the original conversation; model/tool messages remain separate and tool results stay hidden. Version-1 files are loaded compatibly and upgraded on their next save. Session IDs are sanitized before becoming filenames.

`/undo` restores the original bytes of files changed through `write_file`, `edit_file`, or `edit_ast` during the current CoreCoder process, and deletes files created by those tools. Multiple edits to one file still return to its first pre-edit state. If a file changed outside CoreCoder after the latest tracked write, normal undo leaves it untouched as a conflict; `/undo force` explicitly overrides that protection. Undo history survives `/reset`, but is not persisted across process restarts or resumed sessions. Arbitrary filesystem side effects from `bash` cannot be guaranteed and are outside the undo set.

### Layered security review

With `Guard` enabled, every tool call passes through non-overridable built-in boundaries; declared read/write scope, network capability, side-effect and baseline-risk validation; a deterministic risk floor; an optional semantic/AI risk reviewer; human confirmation; rate limiting; and JSONL audit. Semantic review can only raise risk. Confirmation displays the command or target, capability scope, side effect, risk, network destination, and elevated network flags. High-risk publishing, remote mutation, infrastructure changes, and destructive restoration require one-time human confirmation and fail closed in non-interactive mode. `always` is offered only for ordinary permission prompts and creates a process-local rule; explicit `/permit` is required to persist ordinary policy. Rules have stable `usr-...`, `ses-...`, `prj-...`, and `sys-...` IDs: `/revoke` can remove only user/session rules, while project and built-in boundaries remain immutable. Every CLI permission mutation is audited. `/security explain` previews the effective rule, capability, risk, and network decision without executing, prompting, writing an audit decision, or consuming a rate-limit slot.

Tool output that may contain external text carries an `[UNTRUSTED_TOOL_OUTPUT ...]` provenance marker. The guard redacts secrets and flags instruction overrides, forged roles, credential requests, and embedded tool commands as `[SECURITY_FINDINGS]`; the content remains evidence, never authority. A finding taints the rest of the turn so later state-changing calls require human confirmation, and provenance survives context compression. Audit files under `~/.corecoder/audit` include rule source, capability scope, risk, confirmation state, a non-reversible argument summary, and an argument digest. `/audit allow|deny|flag|policy|confirmed [1-100] [tool=<name>]` filters today's records; malformed lines are reported and skipped without hiding valid history. A production semantic classifier can be supplied as `Guard(risk_reviewer=...)`; it returns a `RiskAssessment`, and reviewer failure elevates state-changing calls to high risk.

File writes reject system locations, credential directories, live `.env` files, and private-key targets, including credential directories that do not exist yet. Egress defaults to `confirm`: cloud metadata and link-local endpoints are denied, while private networks, unknown hosts, redirects, authenticated requests, and remote mutation require confirmation. `CORECODER_NETWORK_MODE=deny` blocks destinations outside the comma-separated `CORECODER_NETWORK_ALLOWLIST=api.example.com,*.pythonhosted.org`; private and metadata targets cannot be allowlisted.

With `CORECODER_SANDBOX=1`, unavailable Docker blocks Bash execution instead of silently falling back to the host. Container networking defaults to `CORECODER_SANDBOX_NETWORK=none`; set it explicitly to `bridge` to enable container egress. Command inspection cannot prevent DNS rebinding or arbitrary code from opening sockets at runtime, so production deployments still need container network policy, an egress proxy, or host firewall enforcement.

After every completed turn, CoreCoder first writes a durable pending checkpoint and then extracts stable preferences, profiles, project conventions, feedback, verified procedures, and useful task episodes in a background worker. Consecutive unprocessed turns are coalesced, while token-checked acknowledgements prevent an older extraction from deleting newer chat. Startup recovery also runs in that worker, and normal exit never waits for model requests; unfinished work remains pending for a later run. Active memories are retrieved again for every user turn, scoped to the current project where appropriate, and their use/success/failure statistics influence future ranking. Retrieval also searches original evidence, preserving recall when a Chinese request was summarized into English. Pending entries expose their retry count and last extraction error in `/memory`, while repeated failures are quarantined after three attempts. Memories support versioning, evidence, candidate/active/archive/supersede states, an automatically rebuilt `MEMORY.md`, and a cross-process update lock. `/memory show <id>` inspects an entry, `/memory search <query>` searches, `/memory archive <id>` retires an entry, `/memory approve <id>` explicitly activates one, `/memory reflect` retries pending work, and `/memory forget <id>` permanently removes one. `CORECODER_MEMORY=0` disables learning, while `CORECODER_MEMORY_DIR` changes the location.

Procedure memories require a real successful tool execution with exact verification evidence. If the general extraction pass misses a procedure or returns malformed output, an independent constrained pass can still preserve verified execution assets; pending recovery runs this structured pass first. Episode memories require an actual failed tool execution plus an evidence-backed failure or root-cause lesson, and both types remain project-scoped. Requests to run, analyze, or summarize a task are not treated as user/profile/project memory merely because they mention future reuse; verified reusable steps belong in procedure memory. Legacy entries with only this task-request evidence remain readable for audit but are excluded from retrieval. New execution-derived memories start as non-retrievable candidates; a matching validation from a second independent session promotes them to active, while `/memory approve <id>` allows explicit activation. Repeated replay noise is collapsed before reflection. Persistent `.corecoder/permissions.json` changes require explicit user confirmation instead of being silently used to bypass a blocked command. CoreCoder does not automatically generate or install executable Skills from memories; Skill promotion remains an explicit, reviewed operation.

Multi-user deployments can set `CORECODER_TENANT_ID` and `CORECODER_USER_ID`. Memory, user Skills, and Skill outcome telemetry then live in an isolated `tenants/<tenant>/users/<user>` namespace; omitting both variables preserves the original single-user layout. Identifiers are validated against path traversal. Memory reads use a change-aware file cache and retrieval uses an inverted candidate index, avoiding repeated Markdown parsing and full-corpus scoring on every request.

### Skills

Skills are reusable task guidance layered above atomic tools and classified as `atomic`, `workflow`, or `orchestrator`. CoreCoder discovers built-ins from the package, user skills from `~/.corecoder/skills`, and project skills from `.corecoder/skills`; higher scopes override the same skill ID. Each package contains a compact `skill.json` catalog manifest and full `SKILL.md` instructions.

Routing uses progressive “2+1” disclosure. A compact in-memory inverted catalog first merges exact, tag/signature, context, contrastive, and optional semantic recall into a small candidate set. Metadata-only reranking then considers positive and hard-negative examples; tool, input, context, and permission prerequisites; dependency/conflict edges; historical-failure penalties; scope; and prompt cost. Only selected `SKILL.md` files enter model context. A manifest can declare `resource_modes` so only references matching the current mode are read and only that mode's script/asset paths are exposed. By default there is one automatic primary skill and up to two supporting skills, which must be connected through `dependencies` or `composes_with`.

Route decisions are `explicit`, `auto`, `clarify`, or `abstain`. High-confidence, well-separated v2 matches activate automatically; medium-confidence or close alternatives return exactly one clarification without calling the LLM or exposing tools; weak matches load no skill. Use `$skill.id` or `/skill use <id>` to select explicitly, and say `do not use $skill.id` to exclude it for one turn. `/skill explain` shows recall/rerank scores, confidence, task signature, and rejection reasons; `/skill audit` reports missing, cyclic, contradictory, superseded, and high-overlap definitions. Skill tool restrictions can only remove tools. High-risk skills require confirmation before the first state-changing tool, in addition to the normal security guard.

Schema-v2 manifests can progressively add `layer`, structured `signature` dimensions, `requires`, `resource_modes`, `examples.hard_negative`, `examples.contrastive`, `relations`, and `routing` policy while schema v1 remains compatible. Host applications can provide attachment/artifact, input, connected-app, live-app, permission, external-write, risk, intent-mode, and stable rollout-key context. The lifecycle supports `draft → candidate → shadow → canary → active → deprecated`: shadow skills are scored but never activated, canary rollout uses a stable key and `rollout_percent`, `supersedes` redirects obsolete implicit matches, and `SkillManager.transition` validates and audits editable-skill promotion or rollback. `corecoder.skills.evaluate_router` reports positive-case Precision@1, overall accuracy, false activations, missed skills, clarification and override rates, task success, confidence margin, P95 latency, high-risk confirmation, shadow comparison, candidate count, and estimated loaded tokens.

Set `CORECODER_SKILLS=0` to disable routing. `CORECODER_SKILLS_DIR`, `CORECODER_SKILL_TOP_K`, `CORECODER_SKILL_MAX_ACTIVE`, and `CORECODER_SKILL_PROMPT_CHARS` control the user directory and routing budgets. `CORECODER_SKILL_MIN_SCORE` (default `0.24`) is the candidate floor; `CORECODER_SKILL_CLARIFY_CONFIDENCE` (default `0.65`), `CORECODER_SKILL_AUTO_CONFIDENCE` (default `0.82`), and `CORECODER_SKILL_AMBIGUITY_MARGIN` (default `0.12`) control abstention, clarification, and automatic activation.

Selected routes and terminal outcomes are persisted as aggregate counters in `.telemetry.json` inside the user namespace; prompts and tool output are not retained there. Once enough outcomes exist, bounded failure and partial-success penalties immediately influence reranking, while small samples do not change scores. `/skill metrics` displays routes, outcomes, and current penalties.

Large deployments can supply `semantic_recaller(query, limit)` backed by a vector database or ANN service, returning Skill IDs and similarities directly. The legacy per-Skill `semantic_scorer` remains available for small catalogs, while the batch recall path avoids scanning the full catalog.

`/skill evolve <memory-id>` can turn an active project procedure with at least two independent validations into a package under `.corecoder/skills`. The generated package is always a provenance-linked `candidate`, with implicit routing disabled and canary rollout set to zero; generation never activates it. A maintainer must review its procedure, tools, permissions, boundaries, examples, and verification before moving it through `shadow → canary → active`.

Procedure validation now requires both a completed deliverable and satisfied execution boundaries, in addition to exact verification evidence. Runtime failures, unfinished reports, handoff summaries, and credential-file access cannot contribute validation sessions. Pending checkpoints keep the original turn evidence even when the model context is compressed. `/memory show` displays completion-checked session counts; legacy procedure counters are not sufficient for retrieval or evolution, and `/memory approve` does not manufacture validation evidence. Revalidate legacy procedures in two completed independent sessions. `/skill metrics` also shows clarification counts.

The `read_file` and `grep` tools block live `.env`, credential directories, and private-key files before reading; recursive grep skips them. Named `.env.example`, `.env.sample`, and `.env.template` files remain readable. This is a content-tool boundary, not a general filesystem sandbox. Tool-output replacement markers do not prove that the underlying file contains a placeholder.

DeepSeek message preparation preserves returned reasoning and supplies an empty reasoning field for synthetic assistant messages. Compression preserves the current request. An empty, truncated, or handoff-only answer gets one bounded finalization attempt; an unsuccessful retry remains incomplete. Run offline regressions with `python -m pytest -q`; the synthetic live-provider test is opt-in via `CORECODER_LIVE_TESTS=1` and `python -m pytest -q tests/test_provider_live.py`, using the configured DeepSeek endpoint without sending project content.

## Related Projects

If working through CoreCoder was useful, here are a few other tools I've built around agents and LLM systems:

- **[RepoWiki](https://github.com/he-yufeng/RepoWiki)** — dropped into an unfamiliar codebase? It gives you a guided wiki and a where-to-start reading path, a self-hostable DeepWiki alternative.
- **[FindJobs-Agent](https://github.com/he-yufeng/FindJobs-Agent)** — stop sifting job boards by hand: it ranks postings against your resume and runs mock interviews.
- **[ContractGuard](https://github.com/he-yufeng/ContractGuard)** — catch the risky clauses before you sign: it reads contracts and flags the dangerous bits.
- **[GitSense](https://github.com/he-yufeng/GitSense)** — want to contribute to open source? It finds issues worth your time and gauges whether your PR will get merged.
- **[CodeABC](https://github.com/he-yufeng/CodeABC)** — understand any codebase even if you don't code, built for non-programmers.

## Contributing / License

Before you send anything, run `pytest tests/ -q` (86 tests), `ruff check`, and `compileall`, and make sure they're green. MIT licensed: fork it, learn from it, ship something better. A mention of this project is appreciated.

---

By [Yufeng He](https://github.com/he-yufeng), formerly at Moonshot AI (Kimi). I earlier wrote a fairly complete [Claude Code source analysis](https://zhuanlan.zhihu.com/p/1898797658343862272) on Zhihu; this project is its hands-on counterpart: that one walks you through reading it, this one through rebuilding it.

> CoreCoder was formerly named NanoCoder; it was renamed to avoid confusion with [Nano-Collective/nanocoder](https://github.com/Nano-Collective/nanocoder), and old links redirect here automatically.
