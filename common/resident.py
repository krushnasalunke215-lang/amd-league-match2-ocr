"""Keep the model resident: a tiny JSON-lines server and its client.

The harness starts a NEW PROCESS per image, so the model lives in a long-running
server (started by the container CMD) and app.py is a thin client.

Protocol: the client connects, sends one UTF-8 JSON object followed by "\\n",
reads one JSON object followed by "\\n", and the connection closes.

Readiness: the server process loads and warms up the model FIRST and only then
calls `serve()`, which binds the socket. "The socket accepts connections" is
therefore the readiness signal. If loading fails, the server calls
`mark_failed()` so clients stop waiting immediately instead of burning their
per-image budget.

Transport: on Linux (and anything with socket.AF_UNIX) this ALWAYS uses a unix
socket at `sock_path`. Only on a platform without AF_UNIX (the Windows laptop,
for contract tests with the fake backend) does it fall back to TCP on
127.0.0.1; then `sock_path` is a plain file holding the port number, written
after the listener is up. The container never exposes a port.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Callable, NoReturn

USE_UNIX = hasattr(socket, "AF_UNIX")
MAX_LINE = 1 << 20  # 1 MiB per message
POLL_S = 0.2


class ServerFailed(RuntimeError):
    """The server reported that it could not load its model."""


def failed_marker(sock_path: str) -> Path:
    return Path(str(sock_path) + ".failed")


def mark_failed(sock_path: str, message: str) -> None:
    failed_marker(sock_path).write_text(message, encoding="utf-8")


def clear_markers(sock_path: str) -> None:
    """Remove a stale socket/port file and failed marker before loading starts."""
    for p in (Path(sock_path), failed_marker(sock_path)):
        try:
            p.unlink()
        except FileNotFoundError:
            pass


def _read_line(conn: socket.socket) -> bytes:
    buf = bytearray()
    while b"\n" not in buf:
        chunk = conn.recv(65536)
        if not chunk:
            break
        buf += chunk
        if len(buf) > MAX_LINE:
            raise ValueError("message too large")
    return bytes(buf.split(b"\n", 1)[0])


def _log(msg: str) -> None:
    print(f"[resident {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def serve(sock_path: str, handler: Callable[[dict], dict], *, conn_timeout: float = 120.0) -> NoReturn:
    """Bind `sock_path` and answer requests forever. Call only once the model is warm.

    `ping` is answered directly (it must work while a request is running);
    every other request goes to `handler` one at a time under a lock, because
    there is one GPU and one model. A handler exception becomes
    {"ok": false, "error": ...}; a bad client never stops the loop.
    """
    lock = threading.Lock()

    def handle(conn: socket.socket) -> None:
        with conn:
            try:
                conn.settimeout(conn_timeout)
                req = json.loads(_read_line(conn).decode("utf-8"))
                if not isinstance(req, dict):
                    raise ValueError("request must be a JSON object")
                if req.get("op") == "ping":
                    resp = {"ok": True, "ready": True}
                else:
                    with lock:
                        resp = handler(req)
            except Exception as e:  # noqa: BLE001 - report every failure to the client
                resp = {"ok": False, "error": repr(e)}
            try:
                conn.sendall((json.dumps(resp, ensure_ascii=False) + "\n").encode("utf-8"))
            except OSError as e:
                _log(f"client went away: {e!r}")

    if USE_UNIX:
        try:
            os.unlink(sock_path)
        except FileNotFoundError:
            pass
        tmp = f"{sock_path}.{os.getpid()}.tmp"
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(tmp)
        os.chmod(tmp, 0o666)
        srv.listen(16)
        os.replace(tmp, sock_path)  # the socket appears at its real path fully listening
        _log(f"listening on unix:{sock_path}")
    else:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.bind(("127.0.0.1", int(os.environ.get("RESIDENT_DEV_PORT", "0"))))
        srv.listen(16)
        port = srv.getsockname()[1]
        Path(sock_path).write_text(str(port), encoding="utf-8")
        _log(f"no AF_UNIX on this platform; laptop-only TCP fallback 127.0.0.1:{port} (port file {sock_path})")

    while True:
        conn, _ = srv.accept()
        threading.Thread(target=handle, args=(conn,), daemon=True).start()


def _connect(sock_path: str) -> socket.socket:
    if USE_UNIX:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            s.connect(sock_path)
        except BaseException:
            s.close()
            raise
        return s
    port = int(Path(sock_path).read_text(encoding="utf-8").strip())
    return socket.create_connection(("127.0.0.1", port), timeout=2.0)


def request(sock_path: str, payload: dict, *, deadline: float) -> dict:
    """Send one request and return the response dict.

    `deadline` is an absolute time.monotonic() value. Until it passes, keep
    trying to connect (the server may still be loading its model). Raises
    ServerFailed if the server marked its load as failed, TimeoutError if the
    deadline passes before a full response arrives.
    """
    while True:
        try:
            conn = _connect(sock_path)
            break
        except (OSError, ValueError):
            marker = failed_marker(sock_path)
            if marker.exists():
                raise ServerFailed(marker.read_text(encoding="utf-8", errors="replace")[:500])
            if time.monotonic() + POLL_S >= deadline:
                raise TimeoutError(f"server at {sock_path} not ready before the deadline")
            time.sleep(POLL_S)
    with conn:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("deadline passed while connecting")
        conn.settimeout(remaining)
        conn.sendall((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
        line = _read_line(conn)  # socket.timeout is a TimeoutError subclass
    if not line:
        raise ConnectionError("server closed the connection without a response")
    resp = json.loads(line.decode("utf-8"))
    if not isinstance(resp, dict):
        raise ValueError("response is not a JSON object")
    return resp


def wait_ready(sock_path: str, timeout_s: float) -> bool:
    """True once the server answers ping within `timeout_s`; False on timeout or failure."""
    try:
        return bool(request(sock_path, {"op": "ping"}, deadline=time.monotonic() + timeout_s).get("ok"))
    except (TimeoutError, ServerFailed, ConnectionError, ValueError, OSError):
        return False
