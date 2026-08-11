"""Tests for AST-precise Python editing."""

import pytest

from corecoder.tools import get_tool


# --- helper --------------------------------------------------------------

def _ast_tool():
    return get_tool("edit_ast")


# --- rename_function -----------------------------------------------------

@pytest.mark.asyncio
async def test_rename_function_definition(tmp_path):
    tool = _ast_tool()
    path = tmp_path / "mod.py"
    path.write_text("def old_name():\n    return 1\n")
    r = await tool.execute(file_path=str(path), operation="rename_function",
                           target="old_name", new_text="new_name")
    assert "Edited" in r
    content = path.read_text()
    assert "def new_name():" in content
    assert "def old_name():" not in content


@pytest.mark.asyncio
async def test_rename_function_and_call_sites(tmp_path):
    tool = _ast_tool()
    path = tmp_path / "mod.py"
    path.write_text("def greet():\n    return 'hi'\n\nprint(greet())\n")
    r = await tool.execute(file_path=str(path), operation="rename_function",
                           target="greet", new_text="hello")
    assert "Edited" in r
    content = path.read_text()
    assert "def hello():" in content
    assert "print(hello())" in content
    assert "greet" not in content


@pytest.mark.asyncio
async def test_rename_function_multiple_call_sites(tmp_path):
    tool = _ast_tool()
    path = tmp_path / "mod.py"
    path.write_text("def f():\n    return 1\n\na = f()\nb = f()\nc = f()\n")
    r = await tool.execute(file_path=str(path), operation="rename_function",
                           target="f", new_text="g")
    assert "Edited" in r
    content = path.read_text()
    # 1 definition + 3 call sites = 4 occurrences of "g()"
    assert content.count("g()") == 4
    assert "f()" not in content


@pytest.mark.asyncio
async def test_rename_function_not_found(tmp_path):
    tool = _ast_tool()
    path = tmp_path / "mod.py"
    path.write_text("def a():\n    pass\n")
    r = await tool.execute(file_path=str(path), operation="rename_function",
                           target="nonexistent", new_text="x")
    assert "could not find" in r.lower()


# --- replace_function ----------------------------------------------------

@pytest.mark.asyncio
async def test_replace_function_body(tmp_path):
    tool = _ast_tool()
    path = tmp_path / "mod.py"
    path.write_text("def calc():\n    x = 1\n    return x\n")
    r = await tool.execute(file_path=str(path), operation="replace_function",
                           target="calc", new_text="    return 42")
    assert "Edited" in r
    content = path.read_text()
    assert "def calc():" in content     # signature preserved
    assert "return 42" in content
    assert "x = 1" not in content       # old body gone


@pytest.mark.asyncio
async def test_replace_function_preserves_indentation(tmp_path):
    tool = _ast_tool()
    path = tmp_path / "mod.py"
    path.write_text("class Foo:\n    def method(self):\n        old = 1\n        return old\n")
    r = await tool.execute(file_path=str(path), operation="replace_function",
                           target="method", new_text="        return 'new'")
    assert "Edited" in r
    content = path.read_text()
    assert "    def method(self):" in content
    assert "        return 'new'" in content


# --- insert_after --------------------------------------------------------

@pytest.mark.asyncio
async def test_insert_after_function(tmp_path):
    tool = _ast_tool()
    path = tmp_path / "mod.py"
    path.write_text("def first():\n    pass\n\ndef second():\n    pass\n")
    r = await tool.execute(file_path=str(path), operation="insert_after",
                           target="first", new_text="\ndef middle():\n    pass\n")
    assert "Edited" in r
    content = path.read_text()
    assert content.index("def first") < content.index("def middle") < content.index("def second")


@pytest.mark.asyncio
async def test_insert_after_class(tmp_path):
    tool = _ast_tool()
    path = tmp_path / "mod.py"
    path.write_text("class A:\n    pass\n\nclass B:\n    pass\n")
    r = await tool.execute(file_path=str(path), operation="insert_after",
                           target="A", new_text="\nclass Inserted:\n    pass\n")
    assert "Edited" in r
    content = path.read_text()
    assert content.index("class A") < content.index("class Inserted") < content.index("class B")


@pytest.mark.asyncio
async def test_insert_after_not_found(tmp_path):
    tool = _ast_tool()
    path = tmp_path / "mod.py"
    path.write_text("def a():\n    pass\n")
    r = await tool.execute(file_path=str(path), operation="insert_after",
                           target="b", new_text="# comment")
    assert "could not find" in r.lower()


# --- error handling ------------------------------------------------------

@pytest.mark.asyncio
async def test_edit_ast_file_not_found(tmp_path):
    tool = _ast_tool()
    r = await tool.execute(file_path=str(tmp_path / "nope.py"), operation="rename_function",
                           target="x", new_text="y")
    assert "not found" in r.lower()


@pytest.mark.asyncio
async def test_edit_ast_non_python_file(tmp_path):
    tool = _ast_tool()
    path = tmp_path / "data.txt"
    path.write_text("hello\n")
    r = await tool.execute(file_path=str(path), operation="rename_function",
                           target="x", new_text="y")
    assert "only supports Python files" in r


@pytest.mark.asyncio
async def test_edit_ast_syntax_error_file(tmp_path):
    tool = _ast_tool()
    path = tmp_path / "broken.py"
    path.write_text("def foo(\n    # missing close paren\n")
    r = await tool.execute(file_path=str(path), operation="rename_function",
                           target="x", new_text="y")
    assert "syntax error" in r.lower()


@pytest.mark.asyncio
async def test_edit_ast_would_not_introduce_syntax_error(tmp_path):
    """The tool must verify the edited file still parses before writing."""
    tool = _ast_tool()
    path = tmp_path / "mod.py"
    original = "def f():\n    return 1\n"
    path.write_text(original)
    # replace with code that has a genuine syntax error: unmatched bracket
    r = await tool.execute(file_path=str(path), operation="replace_function",
                           target="f", new_text="    x = (1 + 2\n    return x")
    assert "syntax error" in r.lower() or "Error" in r
    # original file must be untouched
    assert path.read_text() == original


@pytest.mark.asyncio
async def test_edit_ast_unknown_operation(tmp_path):
    tool = _ast_tool()
    path = tmp_path / "mod.py"
    path.write_text("def f():\n    pass\n")
    r = await tool.execute(file_path=str(path), operation="bogus_op",
                           target="f", new_text="x")
    assert "unknown operation" in r.lower()
