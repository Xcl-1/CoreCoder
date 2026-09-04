"""测试 AST 精准编辑——重命名、替换、插入、语法错误保护

直接运行: python smoke_tests/test_edit_ast.py
"""
import asyncio
import tempfile
from pathlib import Path

# 确保能找到 corecoder 模块
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from corecoder.tools.edit_ast import EditASTTool


async def test():
    tool = EditASTTool()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir) / "sample.py"

        # 1. 重命名函数
        tmp.write_text("def hello():\n    return 1\n\nx = hello()\n", encoding="utf-8")
        await tool.execute(file_path=str(tmp), operation="rename_function", target="hello", new_text="greet")
        print("=== rename_function ===")
        print(tmp.read_text(encoding="utf-8"))

        # 2. 替换函数体
        tmp.write_text("def calc(x):\n    return x * 2\n", encoding="utf-8")
        await tool.execute(file_path=str(tmp), operation="replace_function", target="calc", new_text="return x * 3")
        print("=== replace_function ===")
        print(tmp.read_text(encoding="utf-8"))

        # 3. 插入函数
        tmp.write_text("def one():\n    pass\n\ndef three():\n    pass\n", encoding="utf-8")
        await tool.execute(file_path=str(tmp), operation="insert_after", target="one", new_text="def two():\n    pass")
        print("=== insert_after ===")
        print(tmp.read_text(encoding="utf-8"))

        # 4. 语法错误保护
        result = await tool.execute(
            file_path=str(tmp),
            operation="replace_function",
            target="one",
            new_text="broken @@@ syntax",
        )
        print("语法错误拦截:", "syntax error" in result.lower())
        assert "syntax error" in result.lower(), f"expected syntax-error guard, got: {result}"

    print("\nAll AST tests passed!")


if __name__ == "__main__":
    asyncio.run(test())
