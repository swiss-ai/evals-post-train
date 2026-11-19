"""scripts/gaia2/opensandbox_launcher.py against a fake OpenSandbox sandbox (no server, no
network): needs gaia2-runner and the OpenSandbox SDK (requirements/gaia2-runtime.lock)."""

import http.server
import importlib.util
import io
import json
import os
import subprocess
import tarfile
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

HAS_DEPS = importlib.util.find_spec("gaia2_runner") is not None and importlib.util.find_spec("opensandbox") is not None
if HAS_DEPS:
    from scripts.gaia2 import opensandbox_launcher as osl

IMAGE = "ghcr.io/swiss-ai/gaia2-oc:abc"


class FakeSandbox:
    """Container paths live under a local root; tar, id, mkdir and rm are emulated."""

    def __init__(self, root: Path, **create_kwargs):
        self.id = "sbx-1"
        self.root = root
        self.create_kwargs = create_kwargs
        self.runs = []
        self.files_written = {}
        self.killed = self.closed = False
        self.connection_config = SimpleNamespace(protocol="https")
        self.commands = SimpleNamespace(run=self._run)
        self.files = SimpleNamespace(write_files=self._write_files, read_bytes=self._read_bytes)

    def local(self, path: str) -> Path:
        return self.root / path.lstrip("/")

    def _result(self, code=0, out="", err=""):
        lines = lambda s: [SimpleNamespace(text=line) for line in s.splitlines()]  # noqa: E731
        return SimpleNamespace(exit_code=code, error=None, logs=SimpleNamespace(stdout=lines(out), stderr=lines(err)))

    def _run(self, argv, opts=None):
        self.runs.append((argv, opts))
        if opts is not None and opts.background:
            return self._result()
        if argv[0] == "tar" and argv[1] == "czf":
            archive, _, parent, name = argv[2:]
            source = self.local(parent) / name
            if not source.exists():
                return self._result(2, err=f"tar: {name}: No such file or directory")
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tf:
                if name == ".":
                    for item in sorted(source.iterdir()):
                        tf.add(item, arcname=item.name)
                else:
                    tf.add(source, arcname=name)
            self.files_written[archive] = buf.getvalue()
            return self._result()
        if argv[:2] == ["sh", "-c"] and argv[2].startswith("id -u"):
            return self._result(out="1001\n1002\n")
        if argv[0] == "false":
            return self._result(3, err="nope")
        if argv[0] == "cat":
            return self._result(out=self.local(argv[1]).read_text())
        return self._result()

    def _write_files(self, entries):
        for entry in entries:
            self.files_written[entry.path] = entry.data

    def _read_bytes(self, path):
        return self.files_written[path]

    def get_endpoint(self, port):
        headers = {"OPEN-SANDBOX-API-KEY": "server-key", "X-Route": f"{self.id}:{port}"}
        return SimpleNamespace(
            endpoint=f"server.example/sandboxes/{self.id}/proxy/{port}",
            build_request_headers=lambda config: headers,
        )

    def get_diagnostic_events(self, scope):
        return SimpleNamespace(content="events")

    def get_diagnostic_logs(self, scope):
        return SimpleNamespace(content="logs")

    def kill(self):
        self.killed = True

    def close(self):
        self.closed = True


class FakeForward:
    def __init__(self, base_url, headers):
        self.base_url, self.headers, self.port, self.closed = base_url, headers, 45678, False

    def close(self):
        self.closed = True


@unittest.skipUnless(HAS_DEPS, "gaia2-runner / opensandbox not installed (requirements/gaia2-runtime.lock)")
class LauncherTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "sandbox"
        self.root.mkdir()
        self.sandboxes = []
        env = {"OPENSANDBOX_API_KEY": "k", "OPENSANDBOX_RUN_LABEL": "gaia2-test", "GAIA2_ALLOWED_HOSTS": "api.example,judge.example"}
        patcher = mock.patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)

        def factory(**kwargs):
            sandbox = FakeSandbox(self.root, **kwargs)
            self.sandboxes.append(sandbox)
            return sandbox

        self.launcher = osl.OpenSandboxLauncher(sandbox_factory=factory, forward_factory=FakeForward)
        self.scenario = Path(self.tmp.name) / "scenario_1.json"
        self.scenario.write_text('{"events": []}')

    def launch(self, **kwargs):
        return self.launcher.launch(
            IMAGE,
            str(self.scenario),
            env={"GAIA2_JUDGE_MODEL": "glm", "GAIA2_JUDGE_API_KEY": "judge-key", "GAIA2_PROXY_RELAY_URL": "http://p:1"},
            provider="openai-compat",
            model="chonk",
            api_key="model-key",
            adapter_port=40001,
            gateway_port=40002,
            **kwargs,
        )

    def test_launch(self):
        cid = self.launch()
        sandbox = self.sandboxes[0]
        self.assertEqual(cid, "sbx-1")
        kw = sandbox.create_kwargs
        self.assertEqual(kw["image"], IMAGE)
        self.assertTrue(kw["secure_access"])
        self.assertEqual(kw["metadata"]["sandbox-run"], "gaia2-test")
        self.assertEqual((kw["platform"].os, kw["platform"].arch), ("linux", "amd64"))
        policy = kw["network_policy"]
        self.assertEqual(policy.default_action, "deny")
        self.assertEqual([rule.target for rule in policy.egress], ["api.example", "judge.example"])
        # The scenario lands where podman mounts it, before the entrypoint starts.
        self.assertEqual(sandbox.files_written[osl.SCENARIO_PATH], b'{"events": []}')
        background = [(argv, opts) for argv, opts in sandbox.runs if opts.background]
        self.assertEqual(len(background), 1)
        argv, opts = background[0]
        self.assertIn(osl.ENTRYPOINT, argv[-1])
        self.assertEqual(opts.uid, 0)
        env = opts.envs
        self.assertEqual(env["GAIA2_ADAPTER_PORT"], "40001")
        self.assertEqual(env["OPENCLAW_GATEWAY_URL"], "ws://127.0.0.1:40002")
        self.assertEqual((env["PROVIDER"], env["MODEL"], env["API_KEY"]), ("openai-compat", "chonk", "model-key"))
        self.assertEqual(env["OPENAI_COMPAT_API_KEY"], "model-key")
        self.assertEqual(env["GAIA2_JUDGE_API_KEY"], "judge-key")
        self.assertNotIn("GAIA2_PROXY_RELAY_URL", env)
        # Keys never appear on the command line.
        self.assertNotIn("model-key", json.dumps([a for a, _ in sandbox.runs]))
        index_overcommit = next(i for i, (a, _) in enumerate(sandbox.runs) if osl.OVERCOMMIT_COMMAND in " ".join(a))
        self.assertLess(index_overcommit, sandbox.runs.index(background[0]))
        # The adapter is reached through the server, with the sandbox's headers.
        self.assertEqual(self.launcher.get_host_adapter_port(), 45678)
        forward = self.launcher._boxes[cid].forward
        self.assertEqual(forward.base_url, "https://server.example/sandboxes/sbx-1/proxy/40001")
        self.assertEqual(forward.headers["X-Route"], "sbx-1:40001")

    def test_open_network_without_allowlist(self):
        with mock.patch.dict(os.environ):
            del os.environ["GAIA2_ALLOWED_HOSTS"]
            self.launch()
        self.assertEqual(self.sandboxes[0].create_kwargs["network_policy"].default_action, "allow")

    def test_seed_dirs_and_volumes(self):
        seed = Path(self.tmp.name) / "seed"
        seed.mkdir()
        (seed / "memory.md").write_text("hi")
        self.launch(seed_dirs={"/home/agent": str(seed)})
        sandbox = self.sandboxes[0]
        archive = next(p for p in sandbox.files_written if p.startswith("/tmp/.gaia2-seed-"))
        with tarfile.open(fileobj=io.BytesIO(sandbox.files_written[archive])) as tf:
            self.assertEqual(tf.getnames(), ["memory.md"])
        extract = next(a for a, _ in sandbox.runs if "tar xzf" in " ".join(a))
        self.assertIn("-C /home/agent", extract[-1])
        with self.assertRaises(ValueError):
            self.launch(extra_volumes=("/a:/b",))

    def test_failed_launch_kills_the_sandbox(self):
        self.launcher._forward_factory = mock.Mock(side_effect=RuntimeError("no endpoint"))
        with self.assertRaises(RuntimeError):
            self.launch()
        self.assertTrue(self.sandboxes[0].killed)
        self.assertEqual(self.launcher._boxes, {})

    def test_exec(self):
        cid = self.launch()
        sandbox = self.sandboxes[0]
        (self.root / "var").mkdir()
        (self.root / "var" / "events.jsonl").write_text("{}\n")
        self.assertEqual(self.launcher.exec(cid, ["cat", "/var/events.jsonl"], user="gaia2"), "{}\n")
        argv, opts = sandbox.runs[-1]
        self.assertEqual((opts.uid, opts.gid), (1001, 1002))
        self.launcher.exec(cid, ["cat", "/var/events.jsonl"], user="gaia2")
        self.assertEqual(sum(1 for a, _ in sandbox.runs if a[:2] == ["sh", "-c"] and "id -u" in a[2]), 1)  # cached
        self.launcher.exec(cid, ["cat", "/var/events.jsonl"])
        self.assertEqual(sandbox.runs[-1][1].uid, 0)
        with self.assertRaises(subprocess.CalledProcessError) as ctx:
            self.launcher.exec(cid, ["false"])
        self.assertEqual(ctx.exception.returncode, 3)

    def test_stream_without_result_raises(self):
        cid = self.launch()
        sandbox = self.sandboxes[0]
        sandbox.commands.run = lambda argv, opts=None: SimpleNamespace(
            exit_code=None, error=None, logs=SimpleNamespace(stdout=[], stderr=[])
        )
        with self.assertRaises(RuntimeError):
            self.launcher.exec(cid, ["true"])

    def test_copy_from_like_podman_cp(self):
        cid = self.launch()
        (self.root / "tmp" / "openclaw").mkdir(parents=True)
        (self.root / "tmp" / "openclaw" / "openclaw-1.log").write_text("log")
        (self.root / "tmp" / "trace.jsonl").write_text("trace")
        out = Path(self.tmp.name) / "out"
        out.mkdir()
        # A file to a new path, and into an existing directory.
        self.launcher.copy_from(cid, "/tmp/trace.jsonl", str(out / "trace.jsonl"))
        self.assertEqual((out / "trace.jsonl").read_text(), "trace")
        self.launcher.copy_from(cid, "/tmp/trace.jsonl", str(out))
        self.assertTrue((out / "trace.jsonl").is_file())
        # A directory into an existing one, and its contents with "/.".
        self.launcher.copy_from(cid, "/tmp/openclaw/", str(out))
        self.assertEqual((out / "openclaw" / "openclaw-1.log").read_text(), "log")
        contents = out / "contents"
        contents.mkdir()
        self.launcher.copy_from(cid, "/tmp/openclaw/.", str(contents))
        self.assertEqual((contents / "openclaw-1.log").read_text(), "log")
        with self.assertRaises(subprocess.CalledProcessError):
            self.launcher.copy_from(cid, "/var/gaia2/state/missing.json", str(out / "x.json"))
        # The archive is removed from the sandbox each time.
        self.assertTrue(any(a[:2] == ["rm", "-f"] for a, _ in self.sandboxes[0].runs))

    def test_stop(self):
        cid = self.launch()
        forward = self.launcher._boxes[cid].forward
        with mock.patch.dict(os.environ, {"OPENSANDBOX_LOG_DIR": self.tmp.name}):
            self.launcher.stop(cid)
        sandbox = self.sandboxes[0]
        self.assertTrue(sandbox.killed and sandbox.closed and forward.closed)
        self.assertIn("events", (Path(self.tmp.name) / "sbx-1.log").read_text())
        self.launcher.stop(cid)  # already gone: no error
        with self.assertRaises(RuntimeError):
            self.launcher.exec(cid, ["true"])

    def test_install(self):
        import gaia2_runner.cli as cli

        original = cli.LocalLauncher
        self.addCleanup(setattr, cli, "LocalLauncher", original)
        osl.install()
        config = cli.ScenarioExecutionConfig(
            image=IMAGE, runtime="opensandbox", timeout=1, health_timeout=1, container_env={},
            provider=None, model=None, api_key=None, judge_model="j", judge_provider="openai-compat", judge_base_url=None,
        )
        self.assertIsInstance(config.create_runner(8090).launcher, osl.OpenSandboxLauncher)


class _Upstream(http.server.BaseHTTPRequestHandler):
    seen = []

    def log_message(self, *args):
        pass

    def _reply(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        _Upstream.seen.append((self.command, self.path, dict(self.headers), body))
        payload = json.dumps({"connected": True, "path": self.path}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    do_GET = do_POST = _reply


@unittest.skipUnless(HAS_DEPS, "gaia2-runner / opensandbox not installed (requirements/gaia2-runtime.lock)")
class AdapterForwardTest(unittest.TestCase):
    def setUp(self):
        _Upstream.seen = []
        self.upstream = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
        threading.Thread(target=self.upstream.serve_forever, daemon=True).start()
        self.addCleanup(self.upstream.server_close)
        self.addCleanup(self.upstream.shutdown)
        base = f"http://127.0.0.1:{self.upstream.server_address[1]}/sandboxes/s/proxy/8090"
        self.forward = osl.AdapterForward(base, {"OPEN-SANDBOX-API-KEY": "server-key"})
        self.addCleanup(self.forward.close)

    def test_relays_with_the_sandbox_headers(self):
        import httpx

        url = f"http://127.0.0.1:{self.forward.port}"
        self.assertEqual(httpx.get(f"{url}/status", trust_env=False).json()["path"], "/sandboxes/s/proxy/8090/status")
        resp = httpx.post(f"{url}/notify", json={"message": "hallo"}, trust_env=False)
        self.assertEqual(resp.status_code, 200)
        method, path, headers, body = _Upstream.seen[-1]
        self.assertEqual((method, path), ("POST", "/sandboxes/s/proxy/8090/notify"))
        self.assertEqual(headers["OPEN-SANDBOX-API-KEY"], "server-key")
        self.assertEqual(json.loads(body), {"message": "hallo"})

    def test_wait_for_adapter_polls_through_the_forward(self):
        launcher = osl.OpenSandboxLauncher(sandbox_factory=mock.Mock())
        launcher._boxes["s"] = osl._Box(mock.Mock(id="s"), self.forward)
        launcher.wait_for_adapter("s", timeout=5, interval=0.1)
        self.assertEqual(_Upstream.seen[-1][1], "/sandboxes/s/proxy/8090/health")

    def test_unreachable_upstream_is_a_502(self):
        import httpx

        self.upstream.shutdown()
        self.upstream.server_close()
        resp = httpx.get(f"http://127.0.0.1:{self.forward.port}/status", trust_env=False)
        self.assertEqual(resp.status_code, 502)


if __name__ == "__main__":
    unittest.main()
