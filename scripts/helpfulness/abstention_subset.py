"""Stage the pinned AbstentionBench builder with GPQA excluded before construction."""

import argparse
import ast
import hashlib
import json
from pathlib import Path


REVISION = "af06080e2cbfff73f02a7c33135e5567d6f7ce6a"
PINS = {
    "AbstentionBench.py": "800b11a2a93f2145725b47a006fda852266b9fb95ae25800ccf927c34ce8f595",
    "data.py": "eaa0e16a3bc3a78f84bb34ff87d694664b1bdb0526bf8cedbee3bd61b9bd605d",
    "README.md": "c1d1582f689e7dbf765b9961e7e9cebb0f715a07866ce67f95145d092fce1481",
    "UMWP_indices_answerable.json": "9362b5d655e35c8f31bd0b066cfcf80b7100b93e4894784980721b4a2b7dcc47",
    "kuq_new_categories.csv": "ab6fcdfc22a4e014a8f7f1fee99e61d64a571b8d62804ac7a05ce5b5e68c837f",
    "subsampling-indices.json": "238ce5eea5e4bf3675c75f6d7dc66487edae0cfc5487710f731c5a02d89b4759",
}
COMPONENTS = (
    "alcuna", "bbq", "big_bench_disambiguate", "big_bench_known_unknowns", "coconot",
    "falseqa", "gpqa_abstain", "gsm8k_abstain", "known_unknown_questions", "mediq",
    "mmlu_history_abstain", "mmlu_math_abstain", "moral_choice", "musique", "qaqa",
    "qasper", "situated_qa", "squad2", "umwp", "world_sense",
)
INCLUDED = tuple(name for name in COMPONENTS if name != "gpqa_abstain")
PROTOCOL = "adapted-teacher-abstention-bench-without-gpqa-open-weights-v1"


def metadata(options: dict) -> dict:
    scope = options.get("component_scope")
    if scope is None:
        return {}
    if scope != "without_gpqa":
        raise ValueError("Unsupported AbstentionBench component_scope")
    return {
        "protocol": PROTOCOL,
        "component_scope": {
            "name": "without_gpqa", "included_components": list(INCLUDED),
            "excluded_components": ["gpqa_abstain"], "full_upstream_benchmark": False,
        },
    }


def component_dictionary(source: bytes) -> ast.Dict:
    matches = [
        node.value for node in ast.walk(ast.parse(source.decode("utf-8")))
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict)
        and any(isinstance(target, ast.Name) and target.id == "dataset_name_to_dataset"
                for target in node.targets)
    ]
    if len(matches) != 1:
        raise ValueError("Expected the pinned upstream component constructor dictionary")
    return matches[0]


def without_gpqa(source: bytes) -> bytes:
    if hashlib.sha256(source).hexdigest() != PINS["AbstentionBench.py"]:
        raise ValueError("AbstentionBench builder differs from the pinned original bytes")
    dictionary = component_dictionary(source)
    names = [ast.literal_eval(key) for key in dictionary.keys]
    if set(names) != set(COMPONENTS) or names.count("gpqa_abstain") != 1:
        raise ValueError("Upstream component registry differs from the reviewed 20-component source")
    index = names.index("gpqa_abstain")
    key, value = dictionary.keys[index], dictionary.values[index]
    if (
        not isinstance(value, ast.Call) or not isinstance(value.func, ast.Name)
        or value.func.id != "GPQA" or key.lineno != value.end_lineno
    ):
        raise ValueError("Unexpected upstream GPQA constructor shape")
    lines = source.splitlines(keepends=True)
    result = b"".join(lines[:key.lineno - 1] + lines[value.end_lineno:])
    if set(ast.literal_eval(key) for key in component_dictionary(result).keys) != set(INCLUDED):
        raise ValueError("Adapted builder must retain exactly the other 19 components")
    if any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "GPQA"
           for node in ast.walk(ast.parse(result.decode("utf-8")))):
        raise ValueError("Adapted builder still invokes GPQA")
    return result


def stage(source: Path, output: Path, license_file: Path) -> dict:
    originals = {name: (source / name).read_bytes() for name in PINS}
    for name, content in originals.items():
        if hashlib.sha256(content).hexdigest() != PINS[name]:
            raise ValueError(f"Staged AbstentionBench source differs at {name}")
    license_content = license_file.read_bytes()
    modified = without_gpqa(originals["AbstentionBench.py"])
    output.mkdir(parents=True, mode=0o700)
    for name, content in {**originals, "AbstentionBench.py": modified, "LICENSE": license_content}.items():
        (output / name).write_bytes(content)
    manifest = {
        **metadata({"component_scope": "without_gpqa"}),
        "upstream_repository": "facebook/AbstentionBench", "upstream_revision": REVISION,
        "original_sha256": PINS,
        "staged_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in output.iterdir() if path.is_file()
        },
        "adaptation": "Remove only the eager gpqa_abstain: GPQA() entry before any component construction",
        "upstream_loader_executed": False, "data_export_complete": False,
        "upstream_terms_reviewed": False,
    }
    (output / "subset-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--license-file", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = stage(args.source, args.output, args.license_file)
    except (OSError, ValueError, SyntaxError) as error:
        parser.exit(1, f"AbstentionBench subset staging failed: {error}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
