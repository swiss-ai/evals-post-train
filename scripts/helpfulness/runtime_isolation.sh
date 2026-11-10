#!/bin/bash
# Source before using the scorer venv, including when reusing its manifest.
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
LD_LIBRARY_PATH=$(tr ':' '\n' <<< "${LD_LIBRARY_PATH:-}" \
    | { grep -vE '(^|/)(site|dist)-packages(/|$)|/torch/lib$' || true; } \
    | paste -sd: -)
export LD_LIBRARY_PATH
