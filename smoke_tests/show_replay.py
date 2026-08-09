"""查看最新的回放日志"""
import json
from pathlib import Path

files = sorted(Path('replays').glob('*.jsonl'))
if not files:
    print('没有找到回放日志文件，请先运行: corecoder -p "用 bash 执行 echo test"')
else:
    with open(files[-1], encoding='utf-8') as f:
        for line in f:
            step = json.loads(line)
            print(f'步骤 {step["step"]}:')
            print(f'  工具执行数: {len(step["tool_executions"])}')
            print(f'  耗时: {step["step_duration_ms"]}ms')
            for te in step['tool_executions']:
                print(f'    {te["name"]}: 成功={te["success"]}, 耗时={te["duration_ms"]}ms')
