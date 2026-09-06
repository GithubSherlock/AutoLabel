"""Web 复核服务控制纯函数（v1.0 P5）。

TUI 内一键启动两侧复核服务：端口检测 → 已占用提示「已在运行」；
未占用 → spawn detach（脚本路径 `python3 -u autoXdlabel/web/server.py`，
stdout/stderr DEVNULL）+ 轮询端口就绪 → 输出 URL。

端口 env 与两侧 server 同一读取口径（server.py 自身读 env 决定监听
端口，spawn 侧只做检测/轮询）：AUTOLABEL_PORT(8765，2D 主服务，
含 /3d 复核模式) / AUTOLABEL3D_PORT(8766，3D 独立服务)。
spawn 进程 TUI 退出后存活（即用户所愿——标注复核继续）——/web 输出
注明关闭方式。
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

AUTOLABEL_PORT_ENV = "AUTOLABEL_PORT"
AUTOLABEL3D_PORT_ENV = "AUTOLABEL3D_PORT"
_DEFAULT_PORTS = {"2d": 8765, "3d": 8766}


def port_of(which: str) -> tuple[str, int]:
    """(env 名, 端口)——检测与 spawn 同一 env 读取口径。

    env 值非数字（误配如 "http://x"）回退默认端口——/web 命令绝不因此崩。
    """
    env_name = AUTOLABEL3D_PORT_ENV if which == "3d" else AUTOLABEL_PORT_ENV
    raw = os.environ.get(env_name, "")
    try:
        return env_name, int(raw) if raw else _DEFAULT_PORTS[which]
    except ValueError:
        return env_name, _DEFAULT_PORTS[which]


def port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    """TCP 连通性探测（已占用 = 服务已在运行）。"""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def server_script(which: str) -> Path:
    """两侧 server.py 脚本路径（web/ 无 __init__.py，按脚本路径 spawn 不动包结构）。"""
    if which == "3d":
        import auto3dlabel

        return Path(auto3dlabel.__file__).parent / "web" / "server.py"
    import auto2dlabel

    return Path(auto2dlabel.__file__).parent / "web" / "server.py"


def spawn_web_server(which: str, popen: Any = subprocess.Popen) -> int:
    """detach spawn Web 复核服务，返回 pid（端口就绪由 wait_port 轮询）。"""
    cmd = [sys.executable, "-u", str(server_script(which))]
    proc = popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
    )
    return int(proc.pid)


def wait_port(host: str, port: int, timeout: float = 10.0, poll: float = 0.2) -> bool:
    """轮询端口就绪（spawn 后等待服务绑定）。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if port_open(host, port):
            return True
        time.sleep(poll)
    return False
