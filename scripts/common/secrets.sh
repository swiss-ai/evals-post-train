# shellcheck shell=bash
# secrets.sh - Reads the keys in scripts/secrets.env (gitignored; copy scripts/secrets.env.example).
# Sourced by every runner.
#
# One NAME=value per line; blank lines and # comments are skipped, and quotes around a value are
# stripped. A variable that is already set wins over the file, so `WANDB_API_KEY=... bash ...`
# still overrides it. This only sets shell variables: each runner exports the ones its tools need.

_secrets_file="$(dirname "${BASH_SOURCE[0]}")/../secrets.env"
_secrets_skip='^[[:space:]]*(#|$)'
_secrets_entry='^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)=(.*)$'
_secrets_quoted='^"(.*)"$|^'"'"'(.*)'"'"'$'
if [[ -f "$_secrets_file" ]]; then
    _secrets_n=0
    while IFS= read -r _secrets_line || [[ -n "$_secrets_line" ]]; do
        _secrets_n=$((_secrets_n + 1))
        _secrets_line=${_secrets_line%$'\r'}
        if [[ "$_secrets_line" =~ $_secrets_skip ]]; then
            continue
        fi
        if [[ ! "$_secrets_line" =~ $_secrets_entry ]]; then
            # The line itself is not printed: it may hold a key.
            echo "WARNING: scripts/secrets.env line $_secrets_n is not NAME=value; skipped." >&2
            continue
        fi
        _secrets_name=${BASH_REMATCH[1]}
        _secrets_value=${BASH_REMATCH[2]}
        if [[ "$_secrets_value" =~ $_secrets_quoted ]]; then
            _secrets_value=${BASH_REMATCH[1]}${BASH_REMATCH[2]}
        fi
        if [[ -z "${!_secrets_name:-}" ]]; then
            printf -v "$_secrets_name" '%s' "$_secrets_value"
        fi
    done < "$_secrets_file"
fi
unset _secrets_file _secrets_skip _secrets_entry _secrets_quoted _secrets_n _secrets_line \
    _secrets_name _secrets_value
