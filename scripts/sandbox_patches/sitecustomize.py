"""Auto-imported by Python at interpreter startup for any directory on
PYTHONPATH (the standard `sitecustomize` mechanism) -- run_inspect_eval.sh
adds this directory to PYTHONPATH unconditionally before invoking `inspect
eval`/`inspect eval-set`, so this patch is live for every run through it,
not something a task has to opt into.

Patches inspect_ai.util._sandbox.local.LocalSandboxEnvironment.exec (the
`--sandbox local` provider -- see evals-svc's services/scicode.py for why
aaii/scicode defaults to it) to write an oversized `python -c <code>`
command to a temp file and run that instead, when the code would risk
"OSError: [Errno 7] Argument list too long".

Confirmed as a real failure, not a theoretical one: a live aaii/scicode run
crashed 146 of 195 samples in with exactly that OSError, from
subprocess.Popen -> execve inside LocalSandboxEnvironment.exec. Root cause,
read directly from the installed inspect_evals/scicode/scorer.py: SciCode's
scorer composes each subproblem's generated code by concatenating every
*prior* subproblem's solution in the same main problem (compose_code()),
then runs the whole thing as `sandbox().exec(cmd=["python", "-c", code])` --
so a problem with many subproblems can compose a code string large enough to
blow the kernel's ARG_MAX (combined argv+envp size) on its later
subproblems. inspect_evals/scicode's own default `docker` sandbox never hit
this: `docker exec` sends the command as a JSON payload over the Docker
Engine API, not through a host execve() call, so it was never subject to
this OS-level limit in the first place -- a real difference between the two
sandbox providers, not just a config difference.

Scoped narrowly on purpose:
  - Only LocalSandboxEnvironment.exec, not the `docker` sandbox class at
    all -- a run using --sandbox docker (e.g. plain "scicode" on the local
    docker launcher, which has a real daemon) is untouched.
  - Only rewrites a `["python"|"python3", "-c", <code>]` command whose code
    exceeds _ARG_MAX_THRESHOLD -- every other command (and every small `-c`
    invocation) passes through to the original exec() unchanged, so this is
    a no-op for any task other than SciCode's own multi-subproblem
    compositions.
  - The temp file is written and unlinked around the single exec() call, not
    left behind: SciCode's own generated code is the only thing in it.

Verified directly (not guessed): patched a throwaway inspect-ai install,
confirmed a small `-c` command passes through unchanged and a large one is
rewritten to a temp file whose content matches exactly at the moment the
(faked) subprocess call happens, with the file gone again immediately after.
"""

import functools
import os
import tempfile

import inspect_ai.util._sandbox.local as _local_sandbox

# Conservative headroom under Linux's typical ARG_MAX (getconf ARG_MAX is
# usually 2097152 bytes, shared between argv and envp; a job's own
# environment -- EVAL_* vars, cluster-injected ones, etc. -- eats into that
# budget too, so this threshold is deliberately well below the raw ceiling).
_ARG_MAX_THRESHOLD = 100_000

_orig_exec = _local_sandbox.LocalSandboxEnvironment.exec


@functools.wraps(_orig_exec)
async def _patched_exec(self, cmd, *args, **kwargs):
    if (
        len(cmd) == 3
        and cmd[0] in ("python", "python3")
        and cmd[1] == "-c"
        and isinstance(cmd[2], str)
        and len(cmd[2]) > _ARG_MAX_THRESHOLD
    ):
        fd, path = tempfile.mkstemp(suffix=".py")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(cmd[2])
            return await _orig_exec(self, [cmd[0], path], *args, **kwargs)
        finally:
            os.unlink(path)
    return await _orig_exec(self, cmd, *args, **kwargs)


_local_sandbox.LocalSandboxEnvironment.exec = _patched_exec
