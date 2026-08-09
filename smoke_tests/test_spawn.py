"""测试 v1.0 Spawn 多 Agent 委派——需要 LLM"""
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from corecoder.agent import Agent, AgentRole
from corecoder.llm import LLM
from corecoder.config import Config


async def test_role_prompts():
    """验证每个角色都有 prompt"""
    for role in AgentRole:
        from corecoder.agent import role_prompt
        prompt = role_prompt(role)
        assert prompt, f"{role} missing prompt"
        print(f"OK: {role.value} prompt ({len(prompt)} chars)")


async def test_role_tools():
    """验证角色工具过滤"""
    from corecoder.agent import role_tools
    from corecoder.tools import ALL_TOOLS

    all_names = {t.name for t in ALL_TOOLS}

    executor_tools = {t.name for t in role_tools(AgentRole.EXECUTOR, ALL_TOOLS)}
    assert executor_tools == all_names, f"executor should have all tools: {executor_tools}"

    reviewer_tools = {t.name for t in role_tools(AgentRole.REVIEWER, ALL_TOOLS)}
    assert reviewer_tools == {"read_file", "grep", "glob"}, f"reviewer read-only: {reviewer_tools}"

    researcher_tools = {t.name for t in role_tools(AgentRole.RESEARCHER, ALL_TOOLS)}
    assert researcher_tools == {"read_file", "grep", "glob"}, f"researcher read-only: {researcher_tools}"

    planner_tools = {t.name for t in role_tools(AgentRole.PLANNER, ALL_TOOLS)}
    assert planner_tools == set(), f"planner no tools: {planner_tools}"

    print("PASS: 角色工具过滤正确")


async def test_spawn_researcher():
    """spawn 一个 researcher 子 Agent 探索代码库"""
    config = Config.from_env()
    llm = LLM(model=config.model, api_key=config.api_key, base_url=config.base_url)
    agent = Agent(llm=llm, replay=False)

    print(f"模型: {config.model}")
    print("spawn researcher: 探索 corecoder/tools/ 目录结构")
    print()

    result = await agent.spawn(
        task="用 glob 列出 corecoder/tools/ 下所有 .py 文件，然后用 grep 搜索 'class.*Tool' 列出所有工具类名",
        role=AgentRole.RESEARCHER,
    )

    print(result[:800])
    print()

    agent.close()
    print("PASS: Spawn researcher 完成")


async def test_agent_tool_with_role():
    """通过 AgentTool 使用角色参数"""
    from corecoder.tools.agent import AgentTool

    config = Config.from_env()
    llm = LLM(model=config.model, api_key=config.api_key, base_url=config.base_url)
    agent = Agent(llm=llm, replay=False)

    # Agent.__init__ 已经 wire 了 AgentTool 的 _parent_agent
    agent_tool = agent._tool_by_name.get("agent")
    if not agent_tool:
        print("SKIP: no agent tool found")
        agent.close()
        return

    print("测试 AgentTool.execute(task, role='researcher')")
    result = await agent_tool.execute(
        task="用 read_file 查看 pyproject.toml 的内容，报告项目名称和版本号",
        role="researcher",
    )
    print(result[:600])

    agent.close()
    print("PASS: AgentTool with role")


if __name__ == "__main__":
    print("=== 测试 1: 角色 Prompt ===")
    asyncio.run(test_role_prompts())

    print("\n=== 测试 2: 角色工具过滤 ===")
    asyncio.run(test_role_tools())

    print("\n=== 测试 3: Spawn Researcher (调 LLM) ===")
    asyncio.run(test_spawn_researcher())

    print("\n=== 测试 4: AgentTool with role (调 LLM) ===")
    asyncio.run(test_agent_tool_with_role())

    print("\n所有 Spawn 测试通过!")
