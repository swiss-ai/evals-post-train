"""Fixes to published Harbor tasks, applied to a downloaded copy before the run (see
BENCHMARKS[...]["patch_tasks"] in scripts/harbor/run.py).

Each patch checks that the text it replaces is there, so a changed dataset fails loudly instead
of running unpatched.
"""

import re
from pathlib import Path


def _replace(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if text.count(old) != 1:
        raise ValueError(f"{path}: expected {old!r} once, found it {text.count(old)} times")
    path.write_text(text.replace(old, new))


def _sub_all(paths, pattern: str, repl: str) -> int:
    n = 0
    for path in paths:
        text, k = re.subn(pattern, repl, path.read_text())
        if k:
            path.write_text(text)
            n += k
    return n


def patch_aider_polyglot(dataset_dir: Path) -> None:
    """Run every test of each exercise, as Aider's own benchmark does.

    The exercises are Exercism's, which hide all but the first test until a learner un-hides
    them, and the published verifiers (aider-polyglot@1.0) un-hide them only for most Java tasks:
    Rust's #[ignore] tests, JavaScript's xtest/xit, C++'s tests behind EXERCISM_RUN_ALL_TESTS and
    Java's @Disabled variants other than @Disabled("Remove to run test") never run, so a solution
    that passes the first test scores 1. Aider's benchmark runs `cargo test -- --include-ignored`,
    turns xtest into test, builds C++ with -DEXERCISM_RUN_ALL_TESTS=1 and removes every
    @Disabled(...). Also, a C++ test failure fails the build (its CMakeLists runs the tests as
    part of it), and test.sh then exits without a reward: it writes 0 instead. One exception:
    Rust macros' compile-fail tests stay skipped, as their fixtures are missing from the dataset."""
    tasks = sorted(p for p in dataset_dir.iterdir() if (p / "tests" / "test.sh").is_file())
    if not tasks:
        raise ValueError(f"no tasks in {dataset_dir}")
    for task in tasks:
        language = task.name.split("_")[1]  # polyglot_<language>_<exercise>
        test_sh = task / "tests" / "test.sh"
        if language == "rust":
            # macros' compile_fails_* tests build files under tests/invalid/, which the dataset
            # leaves out: they would fail even for the reference solution.
            skip = " --skip compile_fails_" if task.name == "polyglot_rust_macros" else ""
            _replace(test_sh, "cargo test --verbose\n", f"cargo test --verbose -- --include-ignored{skip}\n")
        elif language == "cpp":
            _replace(test_sh, "if cmake .. && make; then", "if cmake -DEXERCISM_RUN_ALL_TESTS=1 .. && make; then")
            _replace(test_sh, 'echo "CMake build failed"\n        exit 1\n', 'echo "CMake build failed"\n        false\n')
        elif language == "javascript":
            _sub_all((task / "tests").glob("*.spec.js"), r"\bx(test|it)\(", r"\1(")
        elif language == "java":
            _sub_all((task / "tests").rglob("*.java"), r'@Disabled(\("[^"]*"\))?', "")
