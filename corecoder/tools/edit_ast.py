"""AST-precise code editing for Python files.

Replaces fragile string matching with structural AST operations.
Uses Python's built-in ``ast`` module — zero additional dependencies.

Supports:
- ``rename_function``: rename a function/method definition and all call sites
- ``replace_function``: replace a function body while preserving signature
- ``insert_after``: insert code after a specific function/class definition

If the file is not Python or the AST can't be parsed, it falls back with
a clear error suggesting ``edit_file`` instead.
"""

import ast
from pathlib import Path

from ._utils import unified_diff
from .base import Tool
from .edit import _changed_files


class EditASTTool(Tool):
    name = "edit_ast"
    description = (
        "Edit Python code using AST-aware operations. "
        "Safer than edit_file for structural changes like renaming functions "
        "across a file or replacing a function body without breaking indentation. "
        "Supports: rename_function, replace_function, insert_after."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the Python file to edit",
            },
            "operation": {
                "type": "string",
                "enum": ["rename_function", "replace_function", "insert_after"],
                "description": "The AST operation to perform",
            },
            "target": {
                "type": "string",
                "description": (
                    "Target identifier. For rename_function: old function name. "
                    "For replace_function: function name to replace body of. "
                    "For insert_after: function/class name to insert after."
                ),
            },
            "new_text": {
                "type": "string",
                "description": (
                    "For rename_function: the new function name. "
                    "For replace_function: the new function body (indented). "
                    "For insert_after: the code to insert."
                ),
            },
        },
        "required": ["file_path", "operation", "target", "new_text"],
    }

    def _execute_sync(self, file_path: str, operation: str, target: str, new_text: str) -> str:
        try:
            p = Path(file_path).expanduser().resolve()
            if not p.exists():
                return f"Error: {file_path} not found"
            if p.suffix != ".py":
                return f"Error: edit_ast only supports Python files. Use edit_file for {p.suffix} files."

            try:
                original = p.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return f"Error: {file_path} is not a UTF-8 text file"

            # parse source to AST, preserving line-number info
            try:
                tree = ast.parse(original)
            except SyntaxError as e:
                return f"Error: {file_path} has a syntax error — use edit_file instead.\n{e}"

            if operation == "rename_function":
                new_content = _rename_function(tree, original, target, new_text)
            elif operation == "replace_function":
                new_content = _replace_function(tree, original, target, new_text)
            elif operation == "insert_after":
                new_content = _insert_after(tree, original, target, new_text)
            else:
                return f"Error: unknown operation '{operation}'"

            if new_content is None:
                return f"Error: could not find {operation} target '{target}' in {file_path}"

            # verify the edited file still parses
            try:
                ast.parse(new_content)
            except SyntaxError as e:
                return (
                    f"Error: edit would introduce a syntax error. Aborting.\n"
                    f"{e}\n\nTip: check indentation in new_text — it must match the original."
                )

            p.write_text(new_content, encoding="utf-8")
            _changed_files.add(str(p))

            diff = unified_diff(original, new_content, str(p))
            return f"Edited {file_path} via AST ({operation})\n{diff}"
        except Exception as e:
            return f"Error: {e}"


# ---- AST operation helpers -----------------------------------------------


def _find_node(tree: ast.AST, name: str, *types: type) -> ast.AST | None:
    """Find the first AST node matching the given type(s) and name."""
    for node in ast.walk(tree):
        if isinstance(node, types):
            node_name = getattr(node, "name", None)
            if node_name == name:
                return node
    return None


def _find_all_name_usages(tree: ast.AST, name: str) -> list[ast.Name]:
    """Find all ast.Name nodes referencing a given identifier."""
    results = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == name:
            results.append(node)
    return results


def _get_node_lines(original: str, node: ast.AST) -> tuple[int, int]:
    """Get the (start_line, end_line) for a node. end_line is inclusive."""
    start = node.lineno
    end = getattr(node, "end_lineno", None) or start
    # walk children to find the true end
    for child in ast.walk(node):
        if child is node:
            continue
        child_end = getattr(child, "end_lineno", None)
        if child_end and child_end > end:
            end = child_end
    return start, end


def _rename_function(tree: ast.AST, original: str, old_name: str, new_name: str) -> str | None:
    """Rename a function definition and all call sites in the file."""
    func = _find_node(tree, old_name, ast.FunctionDef, ast.AsyncFunctionDef)
    if func is None:
        return None

    lines = original.split("\n")

    # rename the definition
    def_line = func.lineno - 1  # 0-based
    lines[def_line] = lines[def_line].replace(f"def {old_name}(", f"def {new_name}(", 1)

    # rename all call sites, working from bottom to top so line numbers stay valid
    usages = _find_all_name_usages(tree, old_name)
    usages.sort(key=lambda u: (u.lineno, u.col_offset), reverse=True)
    for usage in usages:
        if usage.lineno == func.lineno and isinstance(usage.ctx, ast.Store):
            continue  # skip the definition itself
        uline = usage.lineno - 1
        line = lines[uline]
        # find and replace the exact old_name occurrence in this line
        # col_offset is in bytes for Python 3.8+, characters otherwise
        before = line[:usage.col_offset]
        after = line[usage.col_offset:]
        if after.startswith(old_name):
            after = new_name + after[len(old_name):]
        else:
            # fallback: simple string replacement on this line only
            after = after.replace(old_name, new_name, 1)
        lines[uline] = before + after

    return "\n".join(lines)


def _replace_function(tree: ast.AST, original: str, func_name: str, new_body: str) -> str | None:
    """Replace the body of a function while keeping its signature."""
    func = _find_node(tree, func_name, ast.FunctionDef, ast.AsyncFunctionDef)
    if func is None:
        return None

    lines = original.split("\n")
    _, end_line = _get_node_lines(original, func)

    # determine body indentation from the first existing statement
    if func.body:
        body_start_line = func.body[0].lineno - 1
        body_indent = _get_indent(lines[body_start_line])
    else:
        sig_indent = _get_indent(lines[func.lineno - 1])
        body_indent = sig_indent + "    "

    # indent each line of the new body preserving relative indentation
    body_code_lines = new_body.strip("\n").split("\n")
    new_body_lines = _reindent_lines(body_code_lines, body_indent)

    # keep everything before the function body, insert new body, keep everything after
    body_cut = func.body[0].lineno - 1 if func.body else func.lineno
    result = lines[:body_cut] + new_body_lines + lines[end_line:]
    return "\n".join(result)


def _insert_after(tree: ast.AST, original: str, target: str, code: str) -> str | None:
    """Insert code after the given function/class definition."""
    node = _find_node(
        tree, target, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef
    )
    if node is None:
        return None

    _, end_line = _get_node_lines(original, node)
    lines = original.split("\n")

    # match indentation of the target node
    node_indent = _get_indent(lines[node.lineno - 1])

    # preserve relative indentation within the code block
    code_lines = code.strip("\n").split("\n")
    indented_lines = _reindent_lines(code_lines, node_indent)

    result = lines[:end_line] + indented_lines + lines[end_line:]
    return "\n".join(result)


def _reindent_lines(code_lines: list[str], base_indent: str) -> list[str]:
    """Re-indent lines preserving relative indentation from the first non-empty line."""
    # find the first non-empty line's indentation as baseline
    base_stripped = None
    for ln in code_lines:
        if ln.strip():
            base_stripped = len(_get_indent(ln))
            break

    if base_stripped is None:
        return [base_indent + ln for ln in code_lines]

    result = []
    for ln in code_lines:
        if not ln.strip():
            result.append("")
        else:
            current_indent = len(_get_indent(ln))
            relative = current_indent - base_stripped
            result.append(base_indent + " " * max(0, relative) + ln.strip())
    return result


def _get_indent(line: str) -> str:
    """Extract the leading whitespace from a line."""
    return line[: len(line) - len(line.lstrip())]

