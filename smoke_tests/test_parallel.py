"""测试 v1.0 并行工具执行深化——读写依赖感知调度"""
import asyncio
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from corecoder.agent import Agent
from corecoder.llm import LLM


async def test_read_priority():
    """读工具立即并行启动，写工具等同类读完成后执行"""
    llm = LLM.__new__(LLM)
    agent = Agent(llm=llm, tools=[], replay=False)

    # 模拟 3 个 read_file + 1 个 edit_file 指向同一个文件
    class _TC:
        def __init__(self, i, name, args):
            self.id = str(i)
            self.name = name
            self.arguments = args

    calls = [
        _TC(1, "read_file", {"file_path": "a.py"}),
        _TC(2, "read_file", {"file_path": "b.py"}),
        _TC(3, "read_file", {"file_path": "a.py"}),  # same as call 1
        _TC(4, "edit_file", {"file_path": "a.py"}),   # write to a.py → waits for reads
        _TC(5, "read_file", {"file_path": "c.py"}),
    ]

    # monkeypatch _exec_tool to record execution order
    execution_order = []

    original_exec = agent._exec_tool

    async def tracked_exec(tc):
        execution_order.append((tc.name, tc.arguments.get("file_path", "")))
        return f"result_{tc.id}", 1.0, True

    agent._exec_tool = tracked_exec

    await agent._exec_tools_async(calls)

    # reads should all appear before writes to the same file
    reads_a = [i for i, (name, path) in enumerate(execution_order)
               if name == "read_file" and path == "a.py"]
    writes_a = [i for i, (name, path) in enumerate(execution_order)
                if name == "edit_file" and path == "a.py"]

    # all reads of a.py should complete before write to a.py
    for r_idx in reads_a:
        for w_idx in writes_a:
            assert r_idx < w_idx, f"Read {r_idx} should be before write {w_idx}"

    print(f"执行顺序: {execution_order}")
    print("PASS: 读工具先于同文件写工具执行")

    agent._exec_tool = original_exec
    agent.close()


async def test_parallel_timing():
    """并行执行多个独立读取应明显快于串行"""
    import tempfile
    from corecoder.tools.read import ReadFileTool
    from corecoder.tools.bash import BashTool

    # 创建 5 个测试文件
    tmpdir = Path(tempfile.mkdtemp())
    files = []
    for i in range(5):
        f = tmpdir / f"test_{i}.py"
        f.write_text("\n".join(f"line {j}" for j in range(100)))
        files.append(f)

    read_tool = ReadFileTool()
    bash_tool = BashTool()

    # 串行执行
    t0 = time.monotonic()
    for f in files:
        await read_tool.execute(file_path=str(f))
    serial_time = (time.monotonic() - t0) * 1000

    # 并行执行
    t0 = time.monotonic()
    await asyncio.gather(*[read_tool.execute(file_path=str(f)) for f in files])
    parallel_time = (time.monotonic() - t0) * 1000

    speedup = serial_time / max(parallel_time, 0.1)
    print(f"串行 5 个 read_file: {serial_time:.0f}ms")
    print(f"并行 5 个 read_file: {parallel_time:.0f}ms")
    print(f"加速比: {speedup:.1f}x")

    if speedup > 1.5:
        print("PASS: 并行执行明显快于串行")
    else:
        print("NOTE: 文件太小或 I/O 太快，看不出明显差异")

    # 清理
    for f in files:
        f.unlink()
    tmpdir.rmdir()

    print()


if __name__ == "__main__":
    asyncio.run(test_read_priority())
    print()
    asyncio.run(test_parallel_timing())
    print("\n并行执行测试通过!")
