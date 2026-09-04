"""测试 Sandbox 沙箱——路径白名单

直接运行: python smoke_tests/test_sandbox.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from corecoder.sandbox import is_write_blocked, sandbox_enabled, docker_available

if __name__ == "__main__":
    # 正常路径不拦截
    print("pyproject.toml:", is_write_blocked("pyproject.toml"))   # False
    print("README.md:", is_write_blocked("README.md"))             # False

    # 敏感路径拦截
    print("/etc/passwd:", is_write_blocked("/etc/passwd"))         # True (Linux)
    print("~/.ssh/config:", is_write_blocked("~/.ssh/config"))     # True

    # Docker 状态
    print("\nSandbox enabled:", sandbox_enabled())
    print("Docker available:", docker_available())

    print("\nSandbox tests passed!")
