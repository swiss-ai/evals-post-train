"""Verify isolation diagnostics with synthetic package locations, not benchmarks."""

import importlib.metadata
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.helpfulness.check_runtime_isolation import verify_runtime


class RuntimeIsolationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.prefix = Path(directory.name).resolve()
        (self.prefix / "pyvenv.cfg").write_text("include-system-site-packages = false\n")
        self.torch = SimpleNamespace(
            __version__="2.15.0+cpu", version=SimpleNamespace(cuda=None),
            __file__=str(self.prefix / "lib/python3.12/site-packages/torch/__init__.py"),
        )
        for context in (
            patch("sys.prefix", str(self.prefix)),
            patch("sys.base_prefix", "/container/python"),
            patch("sys.path", [str(self.prefix / "lib/python3.12/site-packages")]),
            patch("site.ENABLE_USER_SITE", False),
            patch("importlib.metadata.distribution", side_effect=importlib.metadata.PackageNotFoundError),
            patch.dict("sys.modules", {"torch": self.torch}),
        ):
            context.start()
            self.addCleanup(context.stop)

    def test_clean_cpu_venv_passes(self):
        self.assertTrue(verify_runtime()["isolated"])

    def test_system_inheritance_is_rejected(self):
        (self.prefix / "pyvenv.cfg").write_text("include-system-site-packages = true\n")
        with self.assertRaisesRegex(ValueError, "disable system"):
            verify_runtime()

    def test_external_package_paths_are_rejected(self):
        with patch("sys.path", ["/usr/local/lib/python3.12/dist-packages"]):
            with self.assertRaisesRegex(ValueError, "leaked"):
                verify_runtime()

    def test_user_packages_are_rejected(self):
        with patch("site.ENABLE_USER_SITE", True):
            with self.assertRaisesRegex(ValueError, "user site"):
                verify_runtime()

    def test_serving_distribution_is_rejected(self):
        with patch("importlib.metadata.distribution", return_value=object()):
            with self.assertRaisesRegex(ValueError, "serving packages"):
                verify_runtime()

    def test_cuda_torch_is_rejected(self):
        self.torch.version.cuda = "13.1"
        with self.assertRaisesRegex(ValueError, "CPU-only"):
            verify_runtime()

    def test_external_torch_is_rejected(self):
        self.torch.__file__ = "/container/site-packages/torch/__init__.py"
        with self.assertRaisesRegex(ValueError, "outside"):
            verify_runtime()


if __name__ == "__main__":
    unittest.main()
