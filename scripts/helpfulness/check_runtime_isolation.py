"""Verify an installed scorer runtime cannot inherit serving packages."""

import importlib.metadata
from pathlib import Path
import site
import sys


def verify_runtime() -> dict:
    prefix = Path(sys.prefix).resolve()
    if sys.prefix == sys.base_prefix:
        raise ValueError("Scorer Python must run inside a virtual environment")
    configuration = (prefix / "pyvenv.cfg").read_text(encoding="utf-8").lower()
    if "include-system-site-packages = false" not in configuration:
        raise ValueError("Scorer venv must disable system site-packages")
    if site.ENABLE_USER_SITE:
        raise ValueError("Scorer user site-packages must be disabled")
    package_paths = [
        Path(entry).resolve() for entry in sys.path
        if entry and any(part in ("site-packages", "dist-packages") for part in Path(entry).parts)
    ]
    if any(not path.is_relative_to(prefix) for path in package_paths):
        raise ValueError("Container or user site-packages leaked into scorer sys.path")
    serving = []
    for package in ("vllm", "sglang", "humming-kernels", "spin", "torchvision", "torchaudio"):
        try:
            importlib.metadata.distribution(package)
        except importlib.metadata.PackageNotFoundError:
            continue
        serving.append(package)
    if serving:
        raise ValueError(f"Unexpected serving packages in isolated scorer: {serving}")
    import torch

    if torch.version.cuda is not None or not torch.__version__.endswith("+cpu"):
        raise ValueError("Scorer requires the pinned CPU-only Torch build")
    if not Path(torch.__file__).resolve().is_relative_to(prefix):
        raise ValueError("Scorer imported Torch from outside its venv")
    return {
        "isolated": True, "prefix": str(prefix),
        "system_site_packages": False, "user_site_packages": False,
        "site_package_paths": [str(path) for path in package_paths],
        "serving_packages": serving, "torch": torch.__version__,
        "torch_file": torch.__file__, "cuda_runtime": torch.version.cuda,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(verify_runtime(), indent=2))
