#!/bin/bash
# Fetch chisel (github.com/jpillora/chisel, MIT) for --model-tunnel (scripts/harbor/model_tunnel.py)
# into CHISEL_DIR, checked against the release's sha256 (pinned here). Run on a login node (needs
# internet); idempotent. Both architectures: arm64 for the GH200 nodes (the client), amd64 for the
# OpenSandbox microVMs on Breithorn (the server, uploaded into every sandbox).
#   CHISEL_DIR=$SCRATCH/chisel bash scripts/harbor/fetch_chisel.sh
# Leaves chisel_<version>_linux_{amd64,arm64} in CHISEL_DIR, the layout --model-tunnel expects.
# Taken from swiss-ai/rl-infra backends/nemo-rl/apertus2/fetch-chisel.sh.
set -euo pipefail
CHISEL_VERSION="1.12.0"  # keep in sync with CHISEL_VERSION in model_tunnel.py
CHISEL_DIR="${CHISEL_DIR:?set CHISEL_DIR}"
declare -A SHA256=(  # chisel_1.12.0_checksums.txt
    [amd64]=f3f180f1d93aa72cce4e6386f98cc06569a0146fbd65eb4423cf83e6434bcfe6
    [arm64]=2ec6152cd2c74fe0146d4d79e4e7aa174521368c56e433d55e023a92ea404ec3
)
mkdir -p "${CHISEL_DIR}"
for arch in amd64 arm64; do
    f="${CHISEL_DIR}/chisel_${CHISEL_VERSION}_linux_${arch}"
    [[ -x "${f}" ]] && continue
    curl -fsSL -o "${f}.gz" \
        "https://github.com/jpillora/chisel/releases/download/v${CHISEL_VERSION}/chisel_${CHISEL_VERSION}_linux_${arch}.gz"
    echo "${SHA256[${arch}]}  ${f}.gz" | sha256sum -c --quiet - || { echo "ERROR: chisel ${arch} checksum mismatch" >&2; rm -f "${f}.gz"; exit 1; }
    gunzip -f "${f}.gz"
    chmod +x "${f}"
done
echo "== chisel ${CHISEL_VERSION} in ${CHISEL_DIR}"
