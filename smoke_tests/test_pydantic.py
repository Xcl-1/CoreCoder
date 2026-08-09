"""测试 1: Pydantic 数据模型"""
from corecoder.models import ToolCall, LLMResponse, StepRecord, ToolExecRecord
import json

tc = ToolCall(id='x', name='bash', arguments={'command': 'echo hi'})
resp = LLMResponse(content='done', tool_calls=[tc])
print('message:', resp.message)
print('序列化:', resp.model_dump_json())
print('包含 message?', 'message' in json.loads(resp.model_dump_json()))  # 应为 False
print('\nPydantic 模型测试通过!')
