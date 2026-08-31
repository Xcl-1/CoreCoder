"""Interactive REPL - the user-facing terminal interface."""

import argparse
import asyncio
import json
import logging
import os
import sys

from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from . import __version__
from .agent import Agent
from .config import Config
from .llm import LLM, LiteLLM
from .memory import MemoryEngine
from .security import Guard, PermissionRule
from .session import list_sessions, load_session, save_session
from .skills import SkillManager

console = Console()
logger = logging.getLogger(__name__)


def _parse_args():
    p = argparse.ArgumentParser(
        prog="corecoder",
        description="Minimal AI coding agent. Works with any OpenAI-compatible LLM.",
    )
    p.add_argument("-m", "--model", help="Model name (default: $CORECODER_MODEL or gpt-5.5)")
    p.add_argument("--base-url", help="API base URL (default: $OPENAI_BASE_URL)")
    p.add_argument("--api-key", help="API key (default: $OPENAI_API_KEY)")
    p.add_argument("-p", "--prompt", help="One-shot prompt (non-interactive mode)")
    p.add_argument("-r", "--resume", metavar="ID", help="Resume a saved session")
    p.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")
    return p.parse_args()


def main():
    # configure root logger: WARNING to console, DEBUG to file if requested
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # bump agent/llm loggers to INFO when CORECODER_DEBUG is set
    if os.getenv("CORECODER_DEBUG"):
        logging.getLogger("corecoder").setLevel(logging.DEBUG)

    args = _parse_args()
    config = Config.from_env()

    # CLI args override env vars
    if args.model:
        config.model = args.model
    if args.base_url:
        config.base_url = args.base_url
    if args.api_key:
        config.api_key = args.api_key

    if not config.api_key:
        console.print("[red bold]No API key found.[/]")
        console.print(
            "Set one of: OPENAI_API_KEY, DEEPSEEK_API_KEY, or CORECODER_API_KEY\n"
            "\nExamples:\n"
            "  # OpenAI\n"
            "  export OPENAI_API_KEY=sk-...\n"
            "\n"
            "  # DeepSeek\n"
            "  export OPENAI_API_KEY=sk-... OPENAI_BASE_URL=https://api.deepseek.com\n"
            "\n"
            "  # Ollama (local)\n"
            "  export OPENAI_API_KEY=ollama OPENAI_BASE_URL=http://localhost:11434/v1 CORECODER_MODEL=qwen2.5-coder\n"
        )
        sys.exit(1)

    llm_cls = LiteLLM if config.provider == "litellm" else LLM
    llm = llm_cls(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )

    # security layer — interactive confirmation callback
    guard = Guard(confirm_callback=_cli_confirm)
    _cli_confirm._guard = guard  # enable "always allow" via callback attribute
    memory = _create_memory_engine(config, llm)
    skills = _create_skill_manager(config)
    agent = Agent(
        llm=llm,
        max_context_tokens=config.max_context_tokens,
        guard=guard,
        memory=memory,
        skills=skills,
        session_id=args.resume,
    )

    # resume saved session
    if args.resume:
        loaded = load_session(args.resume)
        if loaded:
            agent.messages, loaded_model = loaded
            # restore the model from the saved session unless overridden by CLI
            if not args.model:
                agent.llm.model = loaded_model
                config.model = loaded_model
            console.print(f"[green]Resumed session: {args.resume} (model: {agent.llm.model})[/green]")
        else:
            console.print(f"[red]Session '{args.resume}' not found.[/red]")
            sys.exit(1)

    # one-shot mode
    if args.prompt:
        try:
            _run_once(agent, args.prompt)
        finally:
            _save_current_session(agent, config)
            agent.close()
        return

    # interactive REPL
    _repl(agent, config, show_history=bool(args.resume))


def _run_once(agent: Agent, prompt: str):
    """Non-interactive: run one prompt and exit."""
    def on_token(tok):
        print(tok, end="", flush=True)

    def on_tool(name, kwargs):
        console.print(f"\n[dim]> {name}({_brief(kwargs)})[/dim]")

    try:
        asyncio.run(agent.chat(prompt, on_token=on_token, on_tool=on_tool))
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(130)
    except Exception:
        logger.exception("One-shot execution failed")
        sys.exit(1)
    print()


def _repl(agent: Agent, config: Config, show_history: bool = False):
    """Interactive read-eval-print loop."""
    replay_info = ""
    if agent._replay:
        replay_info = f"\nReplay: [dim]{agent._replay.path}[/dim]"
    console.print(Panel(
        f"[bold]CoreCoder[/bold] v{__version__}\n"
        f"Model: [cyan]{config.model}[/cyan]"
        + (f"  Base: [dim]{config.base_url}[/dim]" if config.base_url else "")
        + f"\nSession: [dim]{agent.session_id}[/dim] (auto-save enabled)"
        + replay_info
        + "\nType [bold]/help[/bold] for commands, [bold]Ctrl+C[/bold] to cancel, [bold]quit[/bold] to exit.",
        border_style="blue",
    ))

    if show_history:
        _show_history(agent.messages)

    hist_path = os.path.expanduser("~/.corecoder_history")
    history = FileHistory(hist_path)

    # Enter submits, Escape+Enter inserts a newline (for pasting code blocks etc.)
    kb = KeyBindings()

    @kb.add("enter")
    def _submit(event):
        event.current_buffer.validate_and_handle()

    @kb.add("escape", "enter")
    def _newline(event):
        event.current_buffer.insert_text("\n")

    while True:
        try:
            user_input = pt_prompt(
                "You > ",
                history=history,
                multiline=True,
                key_bindings=kb,
                prompt_continuation="...  ",
            ).strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nBye!")
            break

        if not user_input:
            continue

        # built-in commands
        if user_input.lower() in ("quit", "exit", "/quit", "/exit"):
            break
        if user_input == "/help":
            _show_help()
            continue
        if user_input == "/reset":
            _save_current_session(agent, config)
            learned = agent.learn()
            agent.reset()
            suffix = f" Learned {len(learned)} memories." if learned else ""
            console.print(f"[yellow]Conversation reset.[/yellow]{suffix}")
            continue
        if user_input == "/tokens":
            p = agent.llm.total_prompt_tokens
            c = agent.llm.total_completion_tokens
            line = f"Tokens: [cyan]{p}[/cyan] prompt + [cyan]{c}[/cyan] completion = [bold]{p+c}[/bold] total"
            cost = agent.llm.estimated_cost
            if cost is not None:
                line += f"  (~${cost:.4f})"
            console.print(line)
            continue
        if user_input == "/model" or user_input.startswith("/model "):
            new_model = user_input[7:].strip() if user_input.startswith("/model ") else ""
            if new_model:
                agent.llm.model = new_model
                config.model = new_model
                console.print(f"Switched to [cyan]{new_model}[/cyan]")
            else:
                console.print(f"Current model: [cyan]{config.model}[/cyan]")
            continue
        if user_input == "/compact":
            from .context import estimate_tokens
            before = estimate_tokens(agent.messages)
            compressed = agent.context.maybe_compress(agent.messages, agent.llm)
            after = estimate_tokens(agent.messages)
            if compressed:
                console.print(f"[green]Compressed: {before} → {after} tokens ({len(agent.messages)} messages)[/green]")
            else:
                console.print(f"[dim]Nothing to compress ({before} tokens, {len(agent.messages)} messages)[/dim]")
            continue
        if user_input == "/save":
            sid = _save_current_session(agent, config)
            if sid:
                console.print(f"[green]Session saved: {sid}[/green]")
                console.print(f"Resume with: corecoder -r {sid}")
            else:
                console.print("[dim]Nothing to save yet.[/dim]")
            continue
        if user_input == "/diff":
            changed_files = agent.changes.changed_files
            if not changed_files:
                console.print("[dim]No files modified this session.[/dim]")
            else:
                console.print(f"[bold]Files modified this session ({len(changed_files)}):[/bold]")
                for f in sorted(changed_files):
                    console.print(f"  [cyan]{f}[/cyan]")
            continue
        if user_input in ("/undo", "/undo force"):
            _undo_changes(agent, force=user_input.endswith(" force"))
            continue
        if user_input == "/replay":
            if agent._replay:
                console.print(f"Replay log: [cyan]{agent._replay.path}[/cyan]")
                console.print(f"Steps recorded: [bold]{agent._step_number}[/bold]")
            else:
                console.print("[dim]Replay logging is disabled.[/dim]")
            continue
        if user_input.startswith("/plan"):
            task = user_input[6:].strip() if user_input.startswith("/plan ") else ""
            if not task:
                console.print("[yellow]Usage: /plan <task description>[/yellow]")
                continue
            asyncio.run(_do_plan(agent, task))
            continue
        if user_input == "/sessions":
            sessions = list_sessions()
            if not sessions:
                console.print("[dim]No saved sessions.[/dim]")
            else:
                for s in sessions:
                    console.print(f"  [cyan]{s['id']}[/cyan] ({s['model']}, {s['saved_at']}) {s['preview']}")
            continue
        if user_input == "/memory":
            _show_memory(agent, config)
            continue
        if user_input.startswith("/memory forget "):
            memory_id = user_input[len("/memory forget "):].strip()
            if agent.memory and agent.memory.forget(memory_id):
                console.print(f"[green]Forgot memory: {memory_id}[/green]")
            else:
                console.print(f"[yellow]Memory not found: {memory_id}[/yellow]")
            continue
        if user_input.startswith("/memory show "):
            _show_memory_entry(agent, user_input[len("/memory show "):].strip())
            continue
        if user_input.startswith("/memory search "):
            _search_memory(agent, user_input[len("/memory search "):].strip())
            continue
        if user_input.startswith("/memory archive "):
            _archive_memory(agent, user_input[len("/memory archive "):].strip())
            continue
        if user_input.startswith("/memory approve "):
            _approve_memory(agent, user_input[len("/memory approve "):].strip())
            continue
        if user_input == "/memory reflect":
            _reflect_pending(agent)
            continue
        if user_input == "/skills":
            _show_skills(agent, config)
            continue
        if user_input.startswith("/skill search "):
            _search_skills(agent, user_input[len("/skill search "):].strip())
            continue
        if user_input.startswith("/skill show "):
            _show_skill(agent, user_input[len("/skill show "):].strip())
            continue
        if user_input.startswith("/skill use "):
            _pin_skill(agent, user_input[len("/skill use "):].strip())
            continue
        if user_input.startswith("/skill unuse "):
            _unpin_skill(agent, user_input[len("/skill unuse "):].strip())
            continue
        if user_input == "/skill clear":
            if agent.skills:
                agent.skills.clear_pins()
            console.print("[green]Cleared pinned skills.[/green]")
            continue
        if user_input == "/skill reload":
            if agent.skills:
                agent.skills.reload()
                console.print(f"[green]Reloaded {len(agent.skills.registry)} skills.[/green]")
            else:
                console.print("[yellow]Skills are disabled.[/yellow]")
            continue
        if user_input == "/skill explain":
            _explain_skill_route(agent)
            continue
        if user_input == "/permissions":
            _show_permissions(agent)
            continue
        if user_input.startswith("/permit "):
            _permit_rule(agent, user_input[8:].strip())
            continue
        if user_input.startswith("/deny "):
            _deny_rule(agent, user_input[6:].strip())
            continue
        if user_input == "/audit":
            _show_audit(agent)
            continue

        # an unknown /command shouldn't be sent to the model as a prompt
        if user_input.startswith("/"):
            console.print(f"[yellow]Unknown command: {user_input.split()[0]} (try /help)[/yellow]")
            continue

        # call the agent — new list each iteration, closure captures correctly
        streamed: list[str] = []

        def on_token(tok, _output: list[str] = streamed):
            _output.append(tok)
            print(tok, end="", flush=True)

        def on_tool(name, kwargs):
            console.print(f"\n[dim]> {name}({_brief(kwargs)})[/dim]")

        try:
            response = asyncio.run(agent.chat(user_input, on_token=on_token, on_tool=on_tool))
            if streamed:
                print()  # newline after streamed tokens
            else:
                # response wasn't streamed (came after tool calls)
                console.print(Markdown(response))
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/yellow]")
        except Exception:
            logger.exception("Error in agent chat loop")
            console.print("\n[red]An unexpected error occurred. Set CORECODER_DEBUG=1 for details.[/red]")
        finally:
            _save_current_session(agent, config)

    _save_current_session(agent, config)
    agent.close()


def _show_history(messages: list[dict]) -> None:
    """Render the human-facing portion of a resumed conversation."""
    if not messages:
        return

    summary_prefixes = (
        "[Conversation summary — incremental]",
        "[Hard context reset]",
    )
    summary_acknowledgements = {
        "Understood. I have the full context.",
        "Context restored. Continuing from where we left off.",
    }
    hidden_tool_results = 0

    console.rule("[bold]Previous conversation[/bold]", style="dim")
    for message in messages:
        role = message.get("role", "")
        content = message.get("content")
        if content is None:
            text = ""
        elif isinstance(content, str):
            text = content
        else:
            text = str(content)

        if role == "tool":
            hidden_tool_results += 1
            continue

        if role == "user":
            if text.startswith(summary_prefixes):
                console.print(Panel(
                    Markdown(text),
                    title="[bold yellow]Conversation summary[/bold yellow]",
                    border_style="yellow",
                    padding=(0, 1),
                ))
            elif text:
                # Text keeps user-provided Rich markup literal.
                console.print(Panel(
                    Text(text),
                    title="[bold cyan]You[/bold cyan]",
                    border_style="cyan",
                    padding=(0, 1),
                ))
            continue

        if role != "assistant":
            continue

        if text and text not in summary_acknowledgements:
            console.print(Panel(
                Markdown(text),
                title="[bold green]CoreCoder[/bold green]",
                border_style="green",
                padding=(0, 1),
            ))

        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            name = str(function.get("name") or "unknown")
            raw_arguments = function.get("arguments") or {}
            if isinstance(raw_arguments, str):
                try:
                    arguments = json.loads(raw_arguments)
                except json.JSONDecodeError:
                    arguments = {"arguments": raw_arguments}
            else:
                arguments = raw_arguments
            if not isinstance(arguments, dict):
                arguments = {"arguments": arguments}
            console.print(Text(f"> {name}({_brief(arguments)})", style="dim"))

    if hidden_tool_results:
        console.print(Text(
            f"{hidden_tool_results} tool result(s) hidden from history.",
            style="dim",
        ))
    console.rule(style="dim")


async def _do_plan(agent: Agent, task: str):
    """Generate a plan, show it to the user, and execute on confirmation."""
    from rich.table import Table

    console.print(f"\n[bold]Planning for:[/bold] {task}")
    console.print("[dim]Generating plan...[/dim]\n")

    try:
        plan = await agent.plan(task)
    except Exception:
        logger.exception("Plan generation failed")
        console.print("[red]Failed to generate plan.[/red]")
        return

    # display the plan
    table = Table(title=f"Plan: {plan.goal}", border_style="blue")
    table.add_column("#", style="dim", width=4)
    table.add_column("Action", style="white")
    table.add_column("Tool", style="cyan", width=12)
    table.add_column("Expected", style="dim", width=30)

    for step in plan.steps:
        table.add_row(str(step.id), step.action, step.tool or "-", step.expected)

    console.print(table)

    # ask for confirmation (plain input() — pt_prompt conflicts with asyncio.run)
    try:
        choice = input("\nExecute this plan? [y]es / [n]o / [m]odify: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        console.print("\n[yellow]Plan cancelled.[/yellow]")
        return

    if choice in ("n", "no", ""):
        console.print("[yellow]Plan cancelled.[/yellow]")
        return

    if choice in ("m", "modify"):
        modify = input("Describe changes (or just press Enter to cancel): ").strip()
        if not modify:
            console.print("[yellow]Plan cancelled.[/yellow]")
            return
        # re-plan with the modification request
        console.print("[dim]Re-planning with feedback...[/dim]")
        await _do_plan(agent, f"{task}\n\nUser feedback on previous plan: {modify}")
        return

    # execute the plan step by step
    console.print(f"\n[green]Executing {len(plan.steps)} steps...[/green]\n")
    for step in plan.steps:
        console.print(f"[bold blue]Step {step.id}/{len(plan.steps)}:[/bold blue] {step.action}")

        def on_token(tok):
            print(tok, end="", flush=True)

        def on_tool(name, kwargs):
            console.print(f"\n[dim]> {name}({_brief(kwargs)})[/dim]")

        try:
            asyncio.run(
                agent.chat(
                    f"Execute this single step from the plan: {step.action}\n"
                    f"Suggested tool: {step.tool or 'any'}\n"
                    f"Expected result: {step.expected}",
                    on_token=on_token,
                    on_tool=on_tool,
                )
            )
            print()
        except KeyboardInterrupt:
            console.print("\n[yellow]Step interrupted.[/yellow]")
            if input("Continue with remaining steps? [y/n]: ").strip().lower() not in ("y", "yes"):
                break
        except Exception:
            logger.exception("Plan step failed")
            console.print("\n[red]Step failed.[/red]")
            if input("Continue? [y/n]: ").strip().lower() not in ("y", "yes"):
                break

    console.print("\n[green]Plan complete.[/green]")


def _show_help():
    console.print(Panel(
        "[bold]Commands:[/bold]\n"
        "  /help          Show this help\n"
        "  /reset         Clear conversation history\n"
        "  /model         Show current model\n"
        "  /model <name>  Switch model mid-conversation\n"
        "  /tokens        Show token usage\n"
        "  /compact       Compress conversation context\n"
        "  /diff          Show files modified this session\n"
        "  /undo         Undo tracked file changes from this session\n"
        "  /undo force   Undo even when files changed externally\n"
        "  /replay        Show replay log path\n"
        "  /plan <task>   Generate and execute a structured plan\n"
        "  /save          Save session to disk\n"
        "  /sessions      List saved sessions\n"
        "  /memory        List cross-session memories\n"
        "  /memory forget <id> Delete one memory\n"
        "  /memory show <id> Show one memory\n"
        "  /memory search <q> Search active memories\n"
        "  /memory archive <id> Archive one memory\n"
        "  /memory approve <id> Reactivate one memory\n"
        "  /memory reflect Process pending session reflections\n"
        "  /skills        List discovered skills\n"
        "  /skill search <q> Search active skills\n"
        "  /skill show <id> Show a skill manifest\n"
        "  /skill use <id> Pin a skill for this conversation\n"
        "  /skill unuse <id> Unpin a skill\n"
        "  /skill clear   Clear pinned skills\n"
        "  /skill reload  Rescan skill directories\n"
        "  /skill explain Explain the previous route\n"
        "  /permissions   List security rules\n"
        "  /permit <t> <p> Add an allow rule\n"
        "  /deny <t> <p> Add a deny rule\n"
        "  /audit         Show today's audit log\n"
        "  quit           Exit CoreCoder\n"
        "\n"
        "[bold]Input:[/bold]\n"
        "  Enter          Submit message\n"
        "  Esc+Enter      Insert newline (for pasting code)",
        title="CoreCoder Help",
        border_style="dim",
    ))


def _create_memory_engine(config: Config, llm):
    if not config.memory_enabled:
        return None
    engine = MemoryEngine(
        llm=llm,
        root=config.memory_dir,
        project_path=os.getcwd(),
        top_k=config.memory_top_k,
    )
    recovered = engine.recover_pending()
    if recovered:
        logger.info("Recovered memory from %s interrupted session(s)", recovered)
    return engine


def _undo_changes(agent: Agent, *, force: bool = False) -> None:
    if not len(agent.changes):
        console.print("[dim]No tracked file changes to undo.[/dim]")
        return
    result = agent.changes.undo_all(force=force)
    console.print(
        f"[green]Undo complete:[/green] {len(result.restored)} restored, "
        f"{len(result.deleted)} deleted, "
        f"[yellow]{len(result.conflicts)} conflicts[/yellow], "
        f"[red]{len(result.errors)} errors[/red]."
    )
    for path in result.conflicts:
        console.print(f"  [yellow]Conflict, left unchanged: {path}[/yellow]")
    for error in result.errors:
        console.print(f"  [red]{error}[/red]")
    if result.conflicts and not force:
        console.print("[dim]Review conflicts, then use /undo force only if overwriting them is intended.[/dim]")


def _create_skill_manager(config: Config) -> SkillManager | None:
    if not config.skills_enabled:
        return None
    manager = SkillManager.create(
        project_path=os.getcwd(),
        user_dir=config.skills_dir,
        top_k=config.skill_top_k,
        max_active=config.skill_max_active,
        max_prompt_chars=config.skill_prompt_chars,
    )
    for error in manager.registry.errors:
        logger.warning("Skill discovery: %s", error)
    for override in manager.registry.overrides:
        logger.info("Skill override: %s", override)
    return manager


def _save_current_session(agent: Agent, config: Config) -> str | None:
    """Atomically checkpoint the current conversation under one stable id."""
    if not agent.messages:
        return None
    try:
        session_id = save_session(agent.messages, config.model, agent.session_id)
        agent.session_id = session_id
        agent.checkpoint_memory()
        return session_id
    except (OSError, ValueError):
        logger.warning("Could not auto-save session %s", agent.session_id, exc_info=True)
        return None


def _show_memory(agent: Agent, config: Config):
    if agent.memory is None:
        console.print("Memory: [yellow]disabled[/yellow]")
        console.print(f"Directory: [dim]{config.memory_dir}[/dim]")
        return
    console.print("Memory: [green]enabled[/green]")
    console.print(f"Directory: [dim]{agent.memory.store.root}[/dim]")
    stats = agent.memory.stats()
    console.print(
        f"Memories: [bold]{stats['total']}[/bold] "
        f"([cyan]{stats['active']}[/cyan] active, [yellow]{stats['candidate']}[/yellow] candidate, "
        f"[cyan]{stats['archived']}[/cyan] archived, [dim]{stats['superseded']}[/dim] superseded; "
        f"[cyan]{stats['global']}[/cyan] global, [cyan]{stats['project']}[/cyan] project)"
    )
    pending = agent.memory.pending_status()
    console.print(f"Pending reflections: [bold]{len(pending)}[/bold]")
    for item in pending[:5]:
        line = Text("  pending ")
        line.append(str(item["session_id"]), style="yellow")
        line.append(f" attempts={item['attempts']} last_error={item['last_error']}")
        console.print(line)
    for memory in agent.memory.store.list()[:10]:
        line = Text("  ")
        line.append(memory.id, style="cyan")
        line.append(f" [{memory.type}/{memory.scope}/{memory.status}] ")
        line.append(memory.description)
        console.print(line)


def _show_memory_entry(agent: Agent, memory_id: str) -> None:
    if agent.memory is None:
        console.print("[yellow]Memory is disabled.[/yellow]")
        return
    memory = agent.memory.store.get(memory_id)
    if memory is None:
        console.print(f"[yellow]Memory not found: {memory_id}[/yellow]")
        return
    metadata = (
        f"Type: {memory.type}  Scope: {memory.scope}  Status: {memory.status}  Version: {memory.version}\n"
        f"Uses: {memory.use_count}  Success/Failure: {memory.success_count}/{memory.failure_count}\n"
        f"Independent validations: {memory.validation_count}  Last validated: {memory.validated_at or '-'}\n"
        f"Keywords: {', '.join(memory.keywords) or '-'}\n"
        f"Sources: {', '.join(memory.source_sessions) or '-'}"
    )
    console.print(Panel(Markdown(f"# {memory.title}\n\n{memory.content}\n\n---\n\n{metadata}"), border_style="blue"))


def _search_memory(agent: Agent, query: str) -> None:
    if agent.memory is None:
        console.print("[yellow]Memory is disabled.[/yellow]")
        return
    if not query:
        console.print("[yellow]Usage: /memory search <query>[/yellow]")
        return
    matches = agent.memory.search(query)
    if not matches:
        console.print("[dim]No matching active memories.[/dim]")
        return
    for match in matches:
        memory = match.memory
        line = Text("  ")
        line.append(memory.id, style="cyan")
        line.append(f" score={match.score:.4f} [{memory.type}/{memory.scope}] ")
        line.append(memory.description)
        console.print(line)


def _archive_memory(agent: Agent, memory_id: str) -> None:
    if agent.memory is None:
        console.print("[yellow]Memory is disabled.[/yellow]")
        return
    archived = agent.memory.archive(memory_id)
    if archived:
        console.print(f"[green]Archived memory: {archived.id}[/green]")
    else:
        console.print(f"[yellow]Memory not found: {memory_id}[/yellow]")


def _approve_memory(agent: Agent, memory_id: str) -> None:
    if agent.memory is None:
        console.print("[yellow]Memory is disabled.[/yellow]")
        return
    approved = agent.memory.approve(memory_id)
    if approved:
        console.print(f"[green]Approved memory: {approved.id}[/green]")
    else:
        console.print(f"[yellow]Memory not found: {memory_id}[/yellow]")


def _reflect_pending(agent: Agent) -> None:
    if agent.memory is None:
        console.print("[yellow]Memory is disabled.[/yellow]")
        return
    recovered = agent.memory.recover_pending(exclude_session=agent.session_id, force=True)
    pending = agent.memory.pending_status()
    remaining = len(pending)
    console.print(f"Processed [green]{recovered}[/green] pending session(s); [yellow]{remaining}[/yellow] remain.")
    for item in pending[:5]:
        console.print(
            f"[yellow]{item['session_id']}[/yellow] attempts={item['attempts']} "
            f"last_error={item['last_error']}"
        )


# ---- skill helpers -----------------------------------------------------

def _show_skills(agent: Agent, config: Config) -> None:
    if agent.skills is None:
        console.print("Skills: [yellow]disabled[/yellow]")
        console.print(f"Directory: [dim]{config.skills_dir}[/dim]")
        return
    from rich.table import Table
    skills = agent.skills.registry.all(include_inactive=True)
    table = Table(title=f"Skills ({len(skills)})", border_style="blue")
    table.add_column("ID", style="cyan")
    table.add_column("Scope", width=9)
    table.add_column("Status", width=10)
    table.add_column("Version", width=9)
    table.add_column("Summary")
    for skill in skills:
        manifest = skill.manifest
        pinned = " *" if manifest.id in agent.skills.pinned else ""
        table.add_row(
            manifest.id + pinned,
            skill.scope,
            manifest.status,
            manifest.version,
            manifest.summary,
        )
    console.print(table)
    if agent.skills.registry.errors:
        console.print(f"[yellow]{len(agent.skills.registry.errors)} skill package(s) failed validation.[/yellow]")
    if agent.skills.registry.overrides:
        console.print(f"[dim]{len(agent.skills.registry.overrides)} scoped override(s) applied.[/dim]")


def _show_skill(agent: Agent, skill_id: str) -> None:
    if agent.skills is None:
        console.print("[yellow]Skills are disabled.[/yellow]")
        return
    skill = agent.skills.registry.get(skill_id)
    if skill is None:
        console.print(f"[yellow]Skill not found: {skill_id}[/yellow]")
        return
    manifest = skill.manifest
    details = (
        f"# {manifest.name}\n\n{manifest.summary}\n\n"
        f"- ID: `{manifest.id}`\n"
        f"- Version: `{manifest.version}`\n"
        f"- Scope: `{skill.scope}`\n"
        f"- Status: `{manifest.status}`\n"
        f"- Category: {', '.join(manifest.category) or '-'}\n"
        f"- Tags: {', '.join(manifest.tags) or '-'}\n"
        f"- Required tools: {', '.join(manifest.tools.required) or '-'}\n"
        f"- Forbidden tools: {', '.join(manifest.tools.forbidden) or '-'}\n"
        f"- Path: `{skill.path}`"
    )
    console.print(Panel(Markdown(details), border_style="blue"))


def _search_skills(agent: Agent, query: str) -> None:
    if agent.skills is None:
        console.print("[yellow]Skills are disabled.[/yellow]")
        return
    if not query:
        console.print("[yellow]Usage: /skill search <query>[/yellow]")
        return
    matches = agent.skills.search(query)
    if not matches:
        console.print("[dim]No matching active skills.[/dim]")
        return
    for match in matches:
        console.print(
            f"  [cyan]{match.skill.manifest.id}[/cyan] score={match.score:.4f} "
            f"[{match.skill.scope}] {match.skill.manifest.summary}"
        )


def _pin_skill(agent: Agent, skill_id: str) -> None:
    if agent.skills is None:
        console.print("[yellow]Skills are disabled.[/yellow]")
    elif agent.skills.pin(skill_id):
        console.print(f"[green]Pinned skill: {skill_id}[/green]")
    else:
        console.print(f"[yellow]Active skill not found: {skill_id}[/yellow]")


def _unpin_skill(agent: Agent, skill_id: str) -> None:
    if agent.skills is not None and agent.skills.unpin(skill_id):
        console.print(f"[green]Unpinned skill: {skill_id}[/green]")
    else:
        console.print(f"[yellow]Skill was not pinned: {skill_id}[/yellow]")


def _explain_skill_route(agent: Agent) -> None:
    if agent.skills is None:
        console.print("[yellow]Skills are disabled.[/yellow]")
        return
    result = agent.skills.last_result
    if result is None:
        console.print("[dim]No skill route has run in this session.[/dim]")
        return
    console.print(f"[bold]Selected:[/bold] {', '.join(result.selected_ids) or '-'}")
    console.print(f"[bold]Prompt characters:[/bold] {len(result.prompt)}")
    for candidate in result.candidates:
        state = "selected" if candidate.skill.manifest.id in result.selected_ids else "candidate"
        reasons = "; ".join(candidate.reasons) or "metadata similarity"
        console.print(
            f"  [cyan]{candidate.skill.manifest.id}[/cyan] {candidate.score:.4f} "
            f"[{state}] {reasons}"
        )
    for reason in result.rejected:
        console.print(f"  [dim]rejected: {reason}[/dim]")


# ---- security helpers --------------------------------------------------

def _cli_confirm(tool_name: str, arguments: dict, reason: str) -> bool | None:
    """Interactive confirmation callback for the Guard.

    Returns True (allow), False (deny once), or None (cancel).
    """
    from rich.table import Table

    summary = _summarise_args(tool_name, arguments)
    table = Table(title="⚠ Security Confirmation Required", border_style="yellow")
    table.add_column("Field", style="dim")
    table.add_column("Value")
    table.add_row("Tool", tool_name)
    table.add_row("Arguments", summary)
    table.add_row("Reason", reason)
    console.print(table)

    try:
        choice = input("\nAllow? [y]es / [n]o / [a]lways yes for this session: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return None

    if choice in ("y", "yes"):
        return True
    if choice in ("a", "always"):
        # add a temporary user rule to skip future confirms
        if hasattr(_cli_confirm, "_guard"):
            pm = _cli_confirm._guard.permissions
            pm.add_user_rule(PermissionRule(
                tool_name=tool_name,
                pattern=".*",
                action="allow",
                reason=f"user allowed during session: {reason}",
                priority=100,
                source="user",
            ))
            console.print("[green]Added allow rule — won't ask again this session.[/green]")
        return True
    return False


def _show_permissions(agent: Agent) -> None:
    """List all security rules in priority order."""
    if agent.guard is None:
        console.print("[dim]Security guard is not active.[/dim]")
        return

    from rich.table import Table
    rules = agent.guard.permissions.list_rules()
    if not rules:
        console.print("[dim]No rules defined.[/dim]")
        return

    table = Table(title="Security Rules", border_style="blue")
    table.add_column("#", style="dim", width=4)
    table.add_column("Tool", style="cyan")
    table.add_column("Pattern", style="white", width=30)
    table.add_column("Action", width=8)
    table.add_column("Source", width=10)
    table.add_column("Reason", style="dim", width=30)

    for i, r in enumerate(rules[:30]):  # cap at 30 for display
        action_style = {"allow": "[green]allow[/green]", "deny": "[red]deny[/red]", "ask": "[yellow]ask[/yellow]"}
        table.add_row(
            str(i + 1), r.tool_name, r.pattern[:28],
            action_style.get(r.action, r.action), r.source, r.reason[:28],
        )
    console.print(table)
    console.print(f"[dim]Total: {len(rules)} rules (showing first 30)[/dim]")


def _permit_rule(agent: Agent, args: str) -> None:
    """Add a user-level allow rule: /permit <tool> <pattern>"""
    if agent.guard is None:
        console.print("[dim]Security guard is not active.[/dim]")
        return
    parts = args.split(None, 1)
    if len(parts) < 2:
        console.print("[yellow]Usage: /permit <tool> <pattern>[/yellow]")
        console.print("Example: /permit bash git push")
        return
    tool, pattern = parts
    rule = PermissionRule(
        tool_name=tool, pattern=pattern, action="allow",
        reason=f"user-granted: {pattern}", priority=100, source="user",
    )
    agent.guard.permissions.add_user_rule(rule)
    console.print(f"[green]Added allow rule: {tool} ~ {pattern}[/green]")


def _deny_rule(agent: Agent, args: str) -> None:
    """Add a user-level deny rule: /deny <tool> <pattern>"""
    if agent.guard is None:
        console.print("[dim]Security guard is not active.[/dim]")
        return
    parts = args.split(None, 1)
    if len(parts) < 2:
        console.print("[yellow]Usage: /deny <tool> <pattern>[/yellow]")
        console.print("Example: /deny bash rm -rf")
        return
    tool, pattern = parts
    rule = PermissionRule(
        tool_name=tool, pattern=pattern, action="deny",
        reason=f"user-denied: {pattern}", priority=100, source="user",
    )
    agent.guard.permissions.add_user_rule(rule)
    console.print(f"[red]Added deny rule: {tool} ~ {pattern}[/red]")


def _show_audit(agent: Agent) -> None:
    """Show a summary of today's audit log."""
    if agent.guard is None:
        console.print("[dim]Security guard is not active.[/dim]")
        return
    import json
    import time as _time
    today = _time.strftime("%Y-%m-%d")
    log_path = agent.guard.audit._dir / f"audit_{today}.jsonl"
    if not log_path.exists():
        console.print("[dim]No audit entries for today.[/dim]")
        return

    entries = []
    try:
        for line in log_path.read_text(encoding="utf-8").strip().split("\n"):
            if line:
                entries.append(json.loads(line))
    except (OSError, json.JSONDecodeError):
        console.print("[red]Could not read audit log.[/red]")
        return

    if not entries:
        console.print("[dim]No audit entries for today.[/dim]")
        return

    allowed = sum(1 for e in entries if e.get("decision") == "allow")
    denied = sum(1 for e in entries if e.get("decision") != "allow")
    console.print(f"[bold]Audit ({today}):[/bold] [green]{allowed} allowed[/green], [red]{denied} denied[/red], {len(entries)} total")

    # show last 10 entries
    from rich.table import Table
    table = Table(title="Recent Entries", border_style="dim")
    table.add_column("Time", style="dim", width=10)
    table.add_column("Tool", style="cyan")
    table.add_column("Decision", width=10)
    table.add_column("Reason", style="dim", width=40)
    for e in entries[-10:]:
        ts = e.get("timestamp", "")[-8:] or ""  # time portion only
        dec = e.get("decision", "?")
        dec_style = f"[green]{dec}[/green]" if dec == "allow" else f"[red]{dec}[/red]"
        table.add_row(ts, e.get("tool_name", ""), dec_style, e.get("reason", "")[:38])
    console.print(table)


def _summarise_args(tool_name: str, arguments: dict, max_len: int = 200) -> str:
    """Build a short human-readable summary of a tool call for display."""
    if tool_name == "bash":
        cmd = arguments.get("command", "")
        return cmd[:max_len]
    file_path = arguments.get("file_path", "")
    if file_path:
        return file_path[:max_len]
    text = " ".join(str(v)[:80] for v in arguments.values())
    return text[:max_len]


def _brief(kwargs: dict, maxlen: int = 80) -> str:
    s = ", ".join(f"{k}={repr(v)[:40]}" for k, v in kwargs.items())
    return s[:maxlen] + ("..." if len(s) > maxlen else "")
