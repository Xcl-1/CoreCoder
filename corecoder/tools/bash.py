"""Shell command execution with safety checks.

Claude Code's BashTool is 1,143 lines. This is the distilled version:
- Output capture with truncation (head+tail preserved)
- Timeout support
- Dangerous command detection (delegated to security.defaults)
- Working directory tracking (cd awareness)
"""

import asyncio
import contextvars
import os
from pathlib import Path

from ..sandbox import wrap_command
from ..security.defaults import check_dangerous
from .base import Tool

# contextvars is the async-compatible replacement for threading.local().
# Each asyncio task carries its own context, so two concurrent bash calls
# never race on the shared cwd — the same guarantee as before.
_cwd_context: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "bash_cwd", default=None
)


class BashTool(Tool):
    name = "bash"
    input_types = ("shell_command",)
    output_type = "process_result"
    permission_scope = "process:execute"
    side_effect = "dynamic"
    network_access = "dynamic"
    declared_risk = "medium"
    description = (
        "Execute a host-shell command and return stdout, stderr, and exit code. "
        "The command already runs in the workspace: never prepend cd, chain commands, "
        "or redirect output. "
        "On Windows this uses cmd.exe: prefer native commands such as dir, avoid "
        "POSIX-only ls and /dev/null, and invoke PowerShell explicitly when needed. "
        "On other hosts use POSIX syntax."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "pattern": r"^[^&;|<>]*$",
                "description": (
                    "One direct command in the existing workspace. Do not use cd or "
                    "shell operators &, ;, |, <, or >."
                ),
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default 120)",
            },
        },
        "required": ["command"],
    }

    async def execute(
        self,
        command: str,
        timeout: int = 120,
        *,
        cwd: str | os.PathLike[str] | None = None,
    ) -> str:
        # safety check — delegated to security.defaults
        warning = check_dangerous(command)
        if warning:
            return f"⚠ Blocked: {warning}\nCommand: {command}\nIf intentional, modify the command to be more specific."

        # use this task's own tracked working directory
        if cwd is None:
            effective_cwd = _cwd_context.get() or os.getcwd()
        else:
            effective_cwd = str(Path(cwd).expanduser().resolve())

        # sandbox wrapping (no-op if CORECODER_SANDBOX is not set)
        try:
            command = wrap_command(command, cwd=effective_cwd)
        except RuntimeError as exc:
            return f"[Security] Blocked: {exc}"

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=effective_cwd,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return f"Error: timed out after {timeout}s"

            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")

            # track cd commands so next command runs in the right place
            if proc.returncode == 0:
                _update_cwd(command, effective_cwd)
            out = stdout
            if stderr:
                out += f"\n[stderr]\n{stderr}"
            if proc.returncode != 0:
                out += f"\n[exit code: {proc.returncode}]"
            # keep head + tail to preserve the most useful info
            if len(out) > 15_000:
                out = (
                    out[:6000]
                    + f"\n\n... truncated ({len(out)} chars total) ...\n\n"
                    + out[-3000:]
                )
            return out.strip() or "(no output)"
        except Exception as e:
            return f"Error running command: {e}"

def _update_cwd(command: str, current_cwd: str):
    """Track directory changes from cd commands, per thread."""
    # walk each cd in a && chain, resolving relative targets against the dir the
    # previous cd landed in (not the original cwd) so `cd a && cd b` ends in a/b
    running = current_cwd
    changed = False
    for part in command.split("&&"):
        part = part.strip()
        if part.startswith("cd "):
            target = part[3:].strip().strip("'\"")
            if target:
                new_dir = os.path.normpath(os.path.join(running, os.path.expanduser(target)))
                if os.path.isdir(new_dir):
                    running = new_dir
                    changed = True
    if changed:
        _cwd_context.set(running)
