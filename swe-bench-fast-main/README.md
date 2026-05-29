# swe-bench-fast

An eval harness for [SWE-bench](https://www.swebench.com/SWE-bench/) with native ARM64 support. It takes pre-computed predictions (patches from your agent or model), applies them inside Docker containers, runs the test suites, and grades the results. Same role as `swebench.harness.run_evaluation` in the upstream Python harness, reimplemented as a single Go binary.

swe-bench-fast does not generate predictions. It only scores them.

## One-command eval

swe-bench-fast pulls pre-built images from Docker Hub and runs them natively on your host architecture — ARM64 or x86. On ARM64, 78% of instances run natively; the rest use Epoch x86 images via QEMU. No image builds required.

```bash
swe-bench-fast run --dataset swe-bench-full.jsonl --predictions preds.jsonl
```

Pre-built images (ARM64 + x86, multi-arch): [Docker Hub](https://hub.docker.com/repository/docker/greynewell/swe-bench-fast/general)

## 6.3x test runner speedup on ARM64

| Instance | ARM64 native (s) | x86 emulated (s) | Speedup |
|---|---|---|---|
| django__django-13346 | 2.7 | 18.9 | 7.0x |
| matplotlib__matplotlib-14623 | 38.0 | 265.7 | 7.0x |
| scikit-learn__scikit-learn-25102 | 2.7 | 18.2 | 6.6x |
| pytest-dev__pytest-6197 | 4.7 | 28.2 | 6.1x |
| sympy__sympy-11618 | 1.9 | 8.0 | 4.2x |
| **Total (11 instances)** | **87.3** | **551.7** | **6.3x** |

Full results: [benchmark gist](https://gist.github.com/greynewell/497005bb33641503f1a5874f16578088)

Blog post: [SWE-bench Tests Run 6x Faster on ARM64 with Native Containers](https://greynewell.com/blog/swe-bench-arm64-native-containers-6x-faster/)

## Dataset format

JSONL, one instance per line. The official SWE-bench datasets from HuggingFace (`princeton-nlp/SWE-bench`, `princeton-nlp/SWE-bench_Verified`) already use this format.

Required fields: `instance_id`, `repo`, `base_commit`, `version`, `patch`, `test_patch`, `FAIL_TO_PASS`, `PASS_TO_PASS`

## Predictions format

JSONL, one prediction per line. Your agent or model generates these.

```json
{"instance_id": "django__django-13346", "model_name_or_path": "gpt-4", "model_patch": "diff --git a/..."}
```

Required fields: `instance_id` (must match dataset), `model_name_or_path`, `model_patch` (unified diff)

## Commands

```bash
# Score predictions against a dataset
swe-bench-fast run --dataset dataset.jsonl --predictions preds.jsonl

# Build native images from scratch (if you want to build locally instead of pulling)
swe-bench-fast build --dataset dataset.jsonl

# Build and push to a registry
swe-bench-fast build --dataset dataset.jsonl --push docker.io/youruser/swe-bench-arm64

# Validate that images exist (locally or on registry) before a full run
swe-bench-fast validate --dataset dataset.jsonl
```

## Image resolution

The runner automatically selects the right image for each instance:

1. **Local image exists?** Use it (from a previous `build`)
2. **Instance requires x86?** Pull from the x86 registry (Epoch by default), run as `linux/amd64`
3. **Registry configured?** Pull from there using the host architecture — multi-arch manifests in `swe-bench-fast` mean ARM64 hosts get `linux/arm64` and x86 hosts get `linux/amd64` automatically
4. **Fallback:** Pull from the x86 registry

With the default config, 1,798 of 2,294 instances pull from `swe-bench-fast`. The remaining 496 (scikit-learn, matplotlib, xarray) are x86-only and pull from the Epoch registry (QEMU on ARM64).

## Configuration

`swe-bench-fast.toml` in the working directory:

```toml
workers = 8                  # parallel eval workers
timeout = 300                # per-instance timeout in seconds
mem_limit = "4g"             # container memory limit
build_workers = 4            # parallel image builds

# Image registries (auto-selected per instance)
arm64_registry = "docker.io/greynewell/swe-bench-fast"
x86_registry = "ghcr.io/epoch-research"
x86_prefix = "swe-bench.eval"
```

All fields can be overridden with `SWE_BENCH_FAST_*` environment variables (e.g. `SWE_BENCH_FAST_WORKERS=16`).

## CI

The repo includes a GitHub Actions workflow (`build-images.yml`) that builds and pushes ARM64 images via matrixed jobs on ARM64 runners. It splits the dataset into configurable chunks, filters out images that already exist on Docker Hub, and pushes each image individually so partial runs are resumable.

## Built with MIST

swe-bench-fast is built on the [MIST stack](https://miststack.dev), eval-driven infrastructure for AI systems.
