"""测试 Plan 模式——生成计划（需要 LLM）
直接运行: python smoke_tests\test_plan.py
或在 REPL 中: /plan 重构 write.py 添加文件大小检查
"""
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from corecoder.agent import Agent
from corecoder.llm import LLM
from corecoder.config import Config


async def main():
    config = Config.from_env()
    llm = LLM(model=config.model, api_key=config.api_key, base_url=config.base_url)
    agent = Agent(llm=llm, replay=False)

    task = "重构 corecoder/tools/write.py，添加写入前的文件大小检查，超过 1MB 拒绝写入"

    print(f"模型: {config.model}")
    print(f"任务: {task}")
    print("\n生成计划中...\n")

    plan = await agent.plan(task)

    print(f"目标: {plan.goal}")
    print(f"步骤数: {len(plan.steps)}")
    print()
    for s in plan.steps:
        print(f"  {s.id}. {s.action}")
        print(f"     工具: {s.tool or '(无)'}  预期: {s.expected}")
    print()

    agent.close()
    print("Plan mode works!")


asyncio.run(main())
