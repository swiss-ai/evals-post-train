#!/usr/bin/env python3
"""Filter a SWE-bench JSONL dataset to exclude instances already pushed to Docker Hub."""

import json
import sys
import urllib.request
import urllib.error


def fetch_existing_tags(repo: str) -> set[str]:
    """Fetch all existing tags from a Docker Hub repository."""
    tags = set()
    url = f"https://hub.docker.com/v2/repositories/{repo}/tags/?page_size=100"
    while url:
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
        except (urllib.error.URLError, json.JSONDecodeError):
            break
        for t in data.get("results", []):
            tags.add(t["name"])
        url = data.get("next")
    return tags


def sanitize_tag(instance_id: str) -> str:
    """Match the tag sanitization in main.go."""
    return instance_id.replace("/", "-").replace("__", "-").replace(".", "-")


def main():
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <input.jsonl> <output.jsonl> <docker-hub-repo>",
              file=sys.stderr)
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    repo = sys.argv[3]

    existing = fetch_existing_tags(repo)
    print(f"Found {len(existing)} existing tags on {repo}")

    kept = 0
    skipped = 0
    with open(input_path) as inp, open(output_path, "w") as out:
        for line in inp:
            obj = json.loads(line)
            tag = sanitize_tag(obj["instance_id"])
            if tag in existing:
                skipped += 1
            else:
                out.write(line)
                kept += 1

    print(f"Kept {kept}, skipped {skipped} already on Docker Hub")


if __name__ == "__main__":
    main()
