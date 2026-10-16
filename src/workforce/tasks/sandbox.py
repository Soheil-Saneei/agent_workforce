"""Run untrusted Python in a separate, resource-limited process with a hard timeout.

Each program runs in a fresh interpreter (``python -I``) inside a throwaway directory with a
minimal environment, CPU/file-size/core limits, and the destructive-function guard from the
official HumanEval harness. This prevents accidents, not determined attacks: run large
experiments inside a container or VM if model outputs are untrusted.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import signal
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

_GUARD = r"""
import builtins, faulthandler, os, platform, resource, shutil, subprocess, sys

def _limit(kind, value):
    try:
        resource.setrlimit(kind, (value, value))
    except (ValueError, OSError):
        pass

_limit(resource.RLIMIT_CPU, {cpu_seconds})
_limit(resource.RLIMIT_CORE, 0)
_limit(resource.RLIMIT_FSIZE, 16 * 1024 * 1024)
if platform.system() != "Darwin":  # RLIMIT_AS is not enforced on macOS
    _limit(resource.RLIMIT_AS, {memory_bytes})

with open("program.py") as _handle:
    _code = compile(_handle.read(), "program.py", "exec")

# Adapted from openai/human-eval execution.reliability_guard (MIT licence).
faulthandler.disable()
if not {script}:
    builtins.exit = None
    builtins.quit = None
os.environ["OMP_NUM_THREADS"] = "1"
for _name in ("kill", "system", "putenv", "remove", "removedirs", "rmdir", "fchdir", "setuid",
              "fork", "forkpty", "killpg", "rename", "renames", "truncate", "replace", "unlink",
              "fchmod", "fchown", "chmod", "chown", "chroot", "lchflags", "lchmod", "lchown",
              "getcwd", "chdir"):
    if hasattr(os, _name):
        setattr(os, _name, None)
shutil.rmtree = None
shutil.move = None
shutil.chown = None
subprocess.Popen = None
for _mod in ("ipdb", "joblib", "resource", "psutil", "tkinter"):
    sys.modules[_mod] = None

exec(_code, {{"__name__": "__main__" if {script} else "__candidate__"}})
"""


@dataclass(frozen=True)
class ExecResult:
    status: str  # "passed" | "failed" | "timeout"
    returncode: int | None
    stdout: str
    stderr: str
    duration_s: float

    @property
    def passed(self) -> bool:
        return self.status == "passed"


class Sandbox:
    def __init__(self, max_workers: int | None = None, memory_mb: int = 2048):
        self.max_workers = max_workers or max(2, (os.cpu_count() or 4) - 1)
        self.memory_mb = memory_mb
        self._semaphores: dict[int, asyncio.Semaphore] = {}

    def _semaphore(self) -> asyncio.Semaphore:
        loop_id = id(asyncio.get_running_loop())
        if loop_id not in self._semaphores:
            self._semaphores[loop_id] = asyncio.Semaphore(self.max_workers)
        return self._semaphores[loop_id]

    async def run(
        self,
        program: str,
        timeout_s: float = 10.0,
        *,
        stdin: str | None = None,
        script: bool = False,
    ) -> ExecResult:
        """Run ``program`` with ``stdin`` as its standard input. ``script`` runs it as a
        standalone script (``__name__ == "__main__"``, ``exit()`` available), as a judge would;
        otherwise it is imported as a module with ``exit`` disabled (the HumanEval guard)."""
        async with self._semaphore():
            return await self._run(program, timeout_s, stdin, script)

    async def _run(
        self, program: str, timeout_s: float, stdin: str | None, script: bool
    ) -> ExecResult:
        workdir = Path(tempfile.mkdtemp(prefix="workforce-sandbox-"))
        try:
            (workdir / "program.py").write_text(program)
            (workdir / "runner.py").write_text(
                _GUARD.format(
                    cpu_seconds=int(timeout_s) + 1,
                    memory_bytes=self.memory_mb * 1024 * 1024,
                    script=script,
                )
            )
            env = {
                "PATH": "/usr/bin:/bin",
                "HOME": str(workdir),
                "PYTHONHASHSEED": "0",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
            start = time.perf_counter()
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-I",
                "-B",
                "runner.py",
                cwd=workdir,
                env=env,
                stdin=asyncio.subprocess.DEVNULL if stdin is None else asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(None if stdin is None else stdin.encode()), timeout_s
                )
            except TimeoutError:
                _kill_group(proc.pid)
                await proc.wait()
                return ExecResult("timeout", None, "", "", time.perf_counter() - start)
            finally:
                _kill_group(proc.pid)  # reap anything the program left behind
            return ExecResult(
                "passed" if proc.returncode == 0 else "failed",
                proc.returncode,
                stdout.decode(errors="replace")[-4000:],
                stderr.decode(errors="replace")[-4000:],
                time.perf_counter() - start,
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


def _kill_group(pid: int) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, signal.SIGKILL)
