#!/usr/bin/env python3
"""Create multi-arch manifest lists in swe-bench-fast from arch-specific registries.

For each instance in the arm64 or x86 datasets, creates an OCI manifest list
in the target registry (swe-bench-fast) pointing to whichever arch images exist.
When both arches are available, the manifest is multi-arch and Docker automatically
pulls the right image based on the host platform.
"""

import argparse
import json
import subprocess
import sys
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed


def fetch_existing_tags(repo: str) -> set[str]:
    """Fetch all existing tags from a Docker Hub repository."""
    tags = set()
    repo = repo.removeprefix("docker.io/")
    url = f"https://hub.docker.com/v2/repositories/{repo}/tags/?page_size=100"
    while url:
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
        except (urllib.error.URLError, json.JSONDecodeError) as e:
            print(f"Warning: failed to fetch tags for {repo}: {e}", file=sys.stderr)
            break
        for t in data.get("results", []):
            tags.add(t["name"])
        url = data.get("next")
    return tags


def sanitize_tag(instance_id: str) -> str:
    """Match the tag sanitization in Go (spec.computeInstanceTag)."""
    return instance_id.replace("/", "-").replace("__", "-").replace(".", "-").lower()


def load_instance_tags(path: str) -> set[str]:
    tags = set()
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    tags.add(sanitize_tag(obj["instance_id"]))
    except FileNotFoundError:
        print(f"Warning: {path} not found, skipping", file=sys.stderr)
    return tags


def create_manifest(tag: str, sources: list[str], target_registry: str) -> tuple[str, bool, str]:
    target = f"{target_registry}:{tag}"
    cmd = ["docker", "buildx", "imagetools", "create", "--tag", target] + sources
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return tag, False, result.stderr.strip()
    return tag, True, ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm64-dataset", required=True, help="Path to swe-bench-arm64.jsonl")
    parser.add_argument("--x86-dataset", required=True, help="Path to swe-bench-x86.jsonl")
    parser.add_argument("--arm64-registry", required=True, help="e.g. docker.io/greynewell/swe-bench-arm64")
    parser.add_argument("--x86-registry", required=True, help="e.g. docker.io/greynewell/swe-bench-x86")
    parser.add_argument("--target-registry", required=True, help="e.g. docker.io/greynewell/swe-bench-fast")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip instances already present in the target registry")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel manifest creation workers (default: 8)")
    args = parser.parse_args()

    print("Fetching existing tags from registries...")
    arm64_tags = fetch_existing_tags(args.arm64_registry)
    x86_tags = fetch_existing_tags(args.x86_registry)
    print(f"  {args.arm64_registry}: {len(arm64_tags)} tags")
    print(f"  {args.x86_registry}:   {len(x86_tags)} tags")

    target_tags: set[str] = set()
    if args.skip_existing:
        target_tags = fetch_existing_tags(args.target_registry)
        print(f"  {args.target_registry}: {len(target_tags)} existing manifests (will skip)")

    arm64_instances = load_instance_tags(args.arm64_dataset)
    x86_instances = load_instance_tags(args.x86_dataset)
    all_instances = arm64_instances | x86_instances
    print(f"\nInstances: {len(arm64_instances)} arm64, {len(x86_instances)} x86, "
          f"{len(all_instances)} total unique")

    # Build work list: for each instance, collect which arch images actually exist
    work: list[tuple[str, list[str]]] = []
    skipped_missing = 0
    skipped_existing = 0

    for tag in sorted(all_instances):
        if args.skip_existing and tag in target_tags:
            skipped_existing += 1
            continue
        sources = []
        if tag in arm64_tags:
            sources.append(f"{args.arm64_registry}:{tag}")
        if tag in x86_tags:
            sources.append(f"{args.x86_registry}:{tag}")
        if not sources:
            skipped_missing += 1
            continue
        work.append((tag, sources))

    print(f"Will create/update {len(work)} manifests "
          f"({skipped_existing} already in target, {skipped_missing} not yet built in any arch)\n")

    if not work:
        print("Nothing to do.")
        return

    ok = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(create_manifest, tag, sources, args.target_registry): tag
            for tag, sources in work
        }
        for future in as_completed(futures):
            tag, success, err = future.result()
            if success:
                ok += 1
                if ok % 100 == 0:
                    print(f"  Progress: {ok}/{len(work)} created...")
            else:
                print(f"  FAILED {tag}: {err}")
                failed += 1

    print(f"\nDone: {ok} created, {failed} failed, "
          f"{skipped_existing} skipped (already in target), "
          f"{skipped_missing} skipped (not yet built)")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
