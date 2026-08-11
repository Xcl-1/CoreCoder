"""获取当前系统用户名的小工具。"""
import getpass
import os
import sys

# 统一以 UTF-8 输出，避免 Windows 控制台 GBK 编码导致中文乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def get_current_username() -> str:
    """返回当前登录的用户名。"""
    return getpass.getuser()


if __name__ == "__main__":
    user = get_current_username()
    print(f"当前用户名: {user}")
    # 补充显示环境变量中的用户名（可能更准确，如 Windows 下的 USERNAME）
    env_user = os.environ.get("USERNAME") or os.environ.get("USER")
    if env_user and env_user != user:
        print(f"环境变量中的用户名: {env_user}")
