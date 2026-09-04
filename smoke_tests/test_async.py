"""测试 2: 异步工具并行执行"""
import asyncio
from corecoder.tools.bash import BashTool
from corecoder.tools.read import ReadFileTool

async def test():
    bash = BashTool()
    read = ReadFileTool()
    results = await asyncio.gather(
        bash.execute(command='echo hello'),
        read.execute(file_path='pyproject.toml', limit=3),
    )
    print('bash:', results[0])
    print('read:', results[1][:200])
    print('\n异步工具测试通过!')

if __name__ == "__main__":
    asyncio.run(test())
