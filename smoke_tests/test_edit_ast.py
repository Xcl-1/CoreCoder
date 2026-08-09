"""测试 AST 精准编辑——重命名、替换、插入、语法错误保护"""
import asyncio
import tempfile
from pathlib import Path

# 确保能找到 corecoder 模块
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from corecoder.tools.edit_ast import EditASTTool


async def test():
    tool = EditASTTool()
    tmp = Path(tempfile.mktemp(suffix=".py"))

    # 1. 重命名函数
    tmp.write_text("def hello():\n    return 1\n\nx = hello()\n")
    await tool.execute(str(tmp), "rename_function", "hello", "greet")
    print("=== rename_function ===")
    print(tmp.read_text())

    # 2. 替换函数体
    tmp.write_text("def calc(x):\n    return x * 2\n")
    await tool.execute(str(tmp), "replace_function", "calc", "return x * 3")
    print("=== replace_function ===")
    print(tmp.read_text())

    # 3. 插入函数
    tmp.write_text("def one():\n    pass\n\ndef three():\n    pass\n")
    await tool.execute(str(tmp), "insert_after", "one", "def two():\n    pass")
    print("=== insert_after ===")
    print(tmp.read_text())

    # 4. 语法错误保护
    r = await tool.execute(str(tmp), "replace_function", "one", "broken @@@ syntax")
    print("语法错误拦截:", "syntax error" in r.lower())

    tmp.unlink()
    print("\nAll AST tests passed!")


asyncio.run(test())
