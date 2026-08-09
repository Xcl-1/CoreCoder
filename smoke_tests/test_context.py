"""测试 v1.0 上下文三层压缩"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from corecoder.context import ContextManager, estimate_tokens


def test_layer1_tool_type_aware_snip():
    """Layer 1: grep 不截断，bash 保留更多"""
    ctx = ContextManager(max_tokens=10000)

    # 模拟 grep 结果——应该不被截断
    grep_output = "\n".join(f"file_{i}.py:{i}: needle" for i in range(50))
    grep_msg = {"role": "tool", "tool_call_id": "g1", "content": grep_output}
    # 模拟前一条 assistant 消息带 tool_call
    assistant = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "g1", "function": {"name": "grep"}}],
    }

    msgs = [assistant, grep_msg]
    changed = ctx._snip_tool_outputs(msgs)
    # grep 不应该被 snipped
    assert not changed, "grep should never be snipped"
    print("PASS: grep 不被截断")

    # 模拟 bash 输出——100行，应该被 snipped
    bash_output = "\n".join(f"[{i}] output line" for i in range(100))
    bash_msg = {"role": "tool", "tool_call_id": "b1", "content": bash_output}
    assistant2 = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "b1", "function": {"name": "bash"}}],
    }
    msgs2 = [assistant2, bash_msg]
    changed2 = ctx._snip_tool_outputs(msgs2)
    assert changed2, "bash with 100 lines should be snipped"
    snipped = msgs2[1]["content"]
    assert "40" not in snipped.splitlines(), "bash snip should keep only 40 head + 40 tail"
    print("PASS: bash 被截断(40+40)")

    # 模拟普通工具输出
    other_output = "\n".join(f"line {i}" for i in range(200))
    other_msg = {"role": "tool", "tool_call_id": "o1", "content": other_output}
    assistant3 = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "o1", "function": {"name": "read_file"}}],
    }
    msgs3 = [assistant3, other_msg]
    changed3 = ctx._snip_tool_outputs(msgs3)
    assert changed3, "read_file with 200 lines should be snipped"
    snipped3 = msgs3[1]["content"]
    lines3 = snipped3.splitlines()
    # 3 head + ... + 3 tail = 7 lines (including the truncation notice)
    assert "snipped" in snipped3
    print("PASS: read_file 被截断(3+3)")

    print()


def test_layer2_incremental_summarize():
    """Layer 2: 增量摘要——只摘要新增部分"""
    ctx = ContextManager(max_tokens=5000)

    # 构造 30 条消息，每轮 2 条
    msgs = []
    for i in range(30):
        msgs.append({"role": "user", "content": f"task {i} " + "x" * 200})
        msgs.append({"role": "assistant", "content": f"done {i} " + "y" * 200})

    tokens_before = estimate_tokens(msgs)
    # 触发增量摘要
    ctx._incremental_summarize(msgs, None, keep_recent=6)
    tokens_after = estimate_tokens(msgs)
    assert tokens_after < tokens_before, f"Expected compression: {tokens_before} → {tokens_after}"
    print(f"PASS: 增量摘要 {tokens_before} → {tokens_after} tokens ({len(msgs)} messages)")


def test_layer25_structured():
    """Layer 2.5: 结构化分层保留"""
    ctx = ContextManager(max_tokens=5000)

    msgs = []
    for i in range(20):
        msgs.append({"role": "user", "content": f"cmd {i} " + "x" * 900})
        msgs.append({"role": "tool", "tool_call_id": f"t{i}", "content": "z" * 900})

    tokens_before = estimate_tokens(msgs)
    changed = ctx._layered_compress(msgs, keep_recent=6)
    tokens_after = estimate_tokens(msgs)
    l25_count = sum(1 for m in msgs if "[L2.5]" in str(m.get("content", "")))
    if changed:
        assert tokens_after < tokens_before
        assert l25_count > 0, "Expected at least some L2.5 markers"
    print(f"PASS: L2.5 分层压缩 {tokens_before} → {tokens_after} tokens ({l25_count} messages compressed)")


def test_layer3_hard_collapse():
    """Layer 3: 硬压缩——保留最后 4 条"""
    ctx = ContextManager(max_tokens=5000)

    msgs = []
    for i in range(50):
        msgs.append({"role": "user", "content": f"msg {i} " + "a" * 500})
        msgs.append({"role": "tool", "tool_call_id": f"t{i}", "content": "b" * 500})

    tokens_before = estimate_tokens(msgs)
    ctx._hard_collapse(msgs, None)
    tokens_after = estimate_tokens(msgs)
    assert tokens_after < tokens_before
    # hard collapse 后应该只有摘要 + 最后几条
    assert "Hard context reset" in msgs[0]["content"]
    print(f"PASS: L3 硬压缩 {tokens_before} → {tokens_after} tokens ({len(msgs)} messages)")


def test_maybe_compress_pipeline():
    """完整管道：自动选择压缩层级"""
    ctx = ContextManager(max_tokens=3000)

    msgs = []
    for i in range(40):
        msgs.append({"role": "user", "content": f"step {i} " + "A" * 300})
        msgs.append({"role": "tool", "tool_call_id": f"t{i}", "content": "B" * 600})

    tokens_before = estimate_tokens(msgs)
    compressed = ctx.maybe_compress(msgs, None)
    tokens_after = estimate_tokens(msgs)

    print(f"PASS: 完整管道 {'压缩了' if compressed else '无需压缩'}")
    print(f"      {tokens_before} → {tokens_after} tokens")


if __name__ == "__main__":
    test_layer1_tool_type_aware_snip()
    test_layer2_incremental_summarize()
    test_layer25_structured()
    test_layer3_hard_collapse()
    test_maybe_compress_pipeline()
    print("\n所有上下文压缩测试通过!")
