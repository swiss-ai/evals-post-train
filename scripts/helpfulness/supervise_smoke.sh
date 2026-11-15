#!/bin/bash

wait_for_smoke_client() {
    local server_pid="$1" client_pid="$2" server_exit
    while kill -0 "$client_pid" 2>/dev/null; do
        if ! kill -0 "$server_pid" 2>/dev/null; then
            if wait "$server_pid"; then
                server_exit=1
            else
                server_exit=$?
            fi
            printf 'Smoke server exited before the client finished (exit %s); inspect server-step.log\n' \
                "$server_exit" >&2
            return "$server_exit"
        fi
        sleep 1
    done
    wait "$client_pid"
}
