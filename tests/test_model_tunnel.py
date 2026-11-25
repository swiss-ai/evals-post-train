import asyncio
import logging
import os
import platform
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.harbor import model_tunnel
from scripts.harbor.model_tunnel import ModelTunnel, ModelTunnelConfig

NODE_ARCH = model_tunnel.ARCH.get(platform.machine(), "amd64")


def fake_chisel(path: Path, record: Path, connect: bool = True) -> None:
    """A stand-in for the node-side client: records its arguments and AUTH, says it connected."""
    line = 'echo "client: Connected (Latency 1ms)"' if connect else 'echo "client: dial failed"'
    path.write_text(f'#!/bin/bash\nprintf "%s\\n" "$@" > {record}\necho "$AUTH" > {record}.auth\n{line}\nexec sleep 30\n')
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


class FakeSandbox:
    def __init__(self, pod_arch: str = "x86_64", health: str = "HTTP/1.0 200 OK", server_rc: int = 0) -> None:
        self.pod_arch, self.health, self.server_rc = pod_arch, health, server_rc
        self.commands: list[tuple[str, dict | None]] = []
        self.uploads: list[tuple[Path, str]] = []

    async def exec(self, command: str, env):
        self.commands.append((command, env))
        if command == "uname -m":
            return 0, self.pod_arch + "\n"
        if "--reverse" in command:
            return self.server_rc, "" if self.server_rc == 0 else "chisel server not listening after 15s:\nboom"
        if command.startswith("bash -c"):
            return 0, self.health + "\n"
        return 0, "server: Listening on http://0.0.0.0:8765\n"

    async def upload(self, source: Path, target: str) -> None:
        self.uploads.append((source, target))

    async def endpoint(self, port: int):
        return f"https://osb.example/sandboxes/abc/port/{port}/", {"OPEN-SANDBOX-API-KEY": "k", "X-Token": "t"}


class ModelTunnelTest(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.dir = Path(self._dir.name)
        self.addCleanup(self._dir.cleanup)
        self.record = self.dir / "client.args"
        self.cfg = ModelTunnelConfig(target="10.0.0.5:8080", chisel_dir=str(self.dir), connect_timeout_s=10)

    def tunnel(self, sandbox: FakeSandbox) -> ModelTunnel:
        return ModelTunnel(self.cfg, sandbox.exec, sandbox.upload, sandbox.endpoint, logging.getLogger("t"))

    def binaries(self, **kwargs) -> None:
        fake_chisel(self.cfg.binary(NODE_ARCH), self.record, **kwargs)
        (self.dir / f"chisel_{model_tunnel.CHISEL_VERSION}_linux_amd64").write_text("pod binary")

    def test_open_starts_both_ends_and_checks_health(self) -> None:
        self.binaries()
        sandbox = FakeSandbox()

        async def run() -> None:
            tunnel = self.tunnel(sandbox)
            await tunnel.open()
            await tunnel.close()

        asyncio.run(run())
        self.assertEqual(sandbox.uploads, [(self.cfg.binary("amd64"), model_tunnel.POD_BINARY)])
        args = self.record.read_text().split("\n")
        self.assertEqual(args[0], "client")
        self.assertIn("X-Token: t", args)
        self.assertEqual(args[-3:-1], ["https://osb.example/sandboxes/abc/port/8765", "R:127.0.0.1:8000:10.0.0.5:8080"])
        # the same credential on both ends, kept out of argv
        auth = self.record.with_name("client.args.auth").read_text().strip()
        server_env = [env for command, env in sandbox.commands if "--reverse" in command][0]
        self.assertEqual(server_env, {"AUTH": auth})
        self.assertTrue(auth.startswith("harbor:"))
        self.assertNotIn(auth, " ".join(args))

    def test_pod_arch_picks_the_binary(self) -> None:
        self.binaries()
        other = "arm64" if NODE_ARCH == "amd64" else "amd64"
        (self.dir / f"chisel_{model_tunnel.CHISEL_VERSION}_linux_{other}").write_text("pod binary")
        sandbox = FakeSandbox(pod_arch="aarch64" if other == "arm64" else "x86_64")
        asyncio.run(self._open_close(sandbox))
        self.assertEqual(sandbox.uploads[0][0], self.cfg.binary(other))

    async def _open_close(self, sandbox: FakeSandbox) -> None:
        tunnel = self.tunnel(sandbox)
        try:
            await tunnel.open()
        finally:
            await tunnel.close()

    def test_missing_binary_names_the_fetch_script(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "fetch_chisel.sh"):
            asyncio.run(self._open_close(FakeSandbox()))

    def test_server_that_does_not_come_up_fails_before_the_client(self) -> None:
        self.binaries()
        with self.assertRaisesRegex(RuntimeError, "did not come up.*boom"):
            asyncio.run(self._open_close(FakeSandbox(server_rc=1)))
        self.assertFalse(self.record.exists())

    def test_client_that_never_connects_fails_with_both_sides_logs(self) -> None:
        self.binaries(connect=False)
        self.cfg = ModelTunnelConfig(target="x:1", chisel_dir=str(self.dir), connect_timeout_s=1)
        with self.assertRaisesRegex(RuntimeError, "did not connect.*dial failed.*sandbox side"):
            asyncio.run(self._open_close(FakeSandbox()))

    def test_failed_health_check_fails_open(self) -> None:
        self.binaries()
        with self.assertRaisesRegex(RuntimeError, "health|GET /health"):
            asyncio.run(self._open_close(FakeSandbox(health="HTTP/1.0 502 Bad Gateway")))

    def test_config_from_env(self) -> None:
        self.assertIsNone(ModelTunnelConfig.from_env({}))
        cfg = ModelTunnelConfig.from_env({
            model_tunnel.TARGET_ENV: "h:1", model_tunnel.CHISEL_DIR_ENV: "/c", model_tunnel.POD_PORT_ENV: "9",
        })
        self.assertEqual((cfg.target, cfg.chisel_dir, cfg.pod_port), ("h:1", "/c", 9))


class PodScriptsTest(unittest.TestCase):
    """The sandbox-side shell scripts, run for real with sh as the exec service runs them."""

    def test_health_check_answers_from_a_listener(self) -> None:
        import http.server
        import threading

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                self.send_response(200 if self.path == "/health" else 404)
                self.end_headers()

            def log_message(self, *args) -> None:
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        result = subprocess.run(
            ["bash", "-c", model_tunnel.health_check_script(server.server_port, "/health")],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("200", result.stdout)

    def test_health_check_without_listener_fails(self) -> None:
        result = subprocess.run(
            ["bash", "-c", model_tunnel.health_check_script(1, "/health", attempts=2)],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("no answer", result.stdout)

    def test_server_start_fails_without_a_binary(self) -> None:
        script = model_tunnel.server_start_script(ModelTunnelConfig(target="x:1", chisel_dir="/c"), wait_s=1)
        if os.path.exists(model_tunnel.POD_BINARY):
            self.skipTest(f"{model_tunnel.POD_BINARY} exists on this machine")
        self.assertEqual(subprocess.run(["sh", "-c", script], capture_output=True, timeout=20).returncode, 1)


if __name__ == "__main__":
    unittest.main()
