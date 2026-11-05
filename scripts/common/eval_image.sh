#!/bin/bash
# Sourced. require_eval_image sets EVAL_IMAGE_TAG to the commit of this checkout, the tag of the
# prebuilt evaluation image (containers/env_eval.toml) that CI publishes for every pushed commit, and
# fails if that image does not exist: scripts and image must come from the same commit. The package
# is public, so an anonymous pull token is enough. EVAL_DRY_RUN=true skips the network check.

EVAL_IMAGE_REPOSITORY="swiss-ai/evals-post-train-eval"

require_eval_image() {
    local commit token
    commit=$(git rev-parse HEAD) || { echo "Error: not in a git checkout; cannot tell which evaluation image to use" >&2; return 1; }
    if [[ "${EVAL_DRY_RUN:-false}" != "true" ]]; then
        token=$(curl -fs "https://ghcr.io/token?scope=repository:$EVAL_IMAGE_REPOSITORY:pull" \
            | sed -n 's/.*"token":"\([^"]*\)".*/\1/p') || token=""
        if [[ -z "$token" ]] || ! curl -fsI -o /dev/null -H "Authorization: Bearer $token" \
            -H "Accept: application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json" \
            "https://ghcr.io/v2/$EVAL_IMAGE_REPOSITORY/manifests/$commit"; then
            echo "Error: no evaluation image ghcr.io/$EVAL_IMAGE_REPOSITORY:$commit." \
                "Push this commit and wait for the 'Evaluation image' workflow (gh run list --workflow eval-image.yml)." >&2
            return 1
        fi
    fi
    export EVAL_IMAGE_TAG="$commit"
}
