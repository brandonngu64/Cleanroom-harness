"""Command execution contracts and a local runner for trusted repositories."""

from collections.abc import Sequence
from dataclasses import dataclass
import math
from pathlib import Path
import subprocess
import tempfile
from time import perf_counter
from typing import BinaryIO, Protocol


@dataclass(frozen=True)
class RunResult:
    """Observed command output, separate from any judgment about task success."""

    command: tuple[str, ...]
    cwd: Path
    exit_code: int | None
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False
    error: str | None = None
    output_truncated: bool = False

    @property
    def succeeded(self) -> bool:
        return (
            self.exit_code == 0
            and not self.timed_out
            and self.error is None
        )


class Runner(Protocol):
    """The execution interface used by the rest of the harness."""

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path,
        timeout: float = 60.0,
    ) -> RunResult:
        ...


class LocalRunner:
    """Run trusted commands locally using the current user's permissions.

    A timeout stops the launched process, not its entire process tree. Output
    is collected in temporary files, so the return limit bounds memory and
    returned text, not temporary disk usage. This runner provides no sandbox
    or CPU/memory limits. Output is decoded as UTF-8 with replacement.
    """

    def __init__(self, max_output_bytes: int = 64 * 1024):
        if (
            not isinstance(max_output_bytes, int)
            or isinstance(max_output_bytes, bool)
            or max_output_bytes <= 0
        ):
            raise ValueError("max_output_bytes must be a positive integer")

        self.max_output_bytes = max_output_bytes

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path,
        timeout: float = 60.0,
    ) -> RunResult:
        if isinstance(command, (str, bytes)):
            raise ValueError("Pass command arguments as a sequence, not a string")

        arguments = tuple(command)
        if (
            not arguments
            or any(not isinstance(argument, str) for argument in arguments)
            or not arguments[0]
            or any("\0" in argument for argument in arguments)
        ):
            raise ValueError(
                "Command must contain a program and valid string arguments"
            )

        if (
            not isinstance(timeout, (int, float))
            or isinstance(timeout, bool)
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("timeout must be a positive, finite number of seconds")

        root = Path(cwd).resolve()
        if not root.is_dir():
            raise ValueError(f"Invalid working directory: {cwd}")

        started = perf_counter()
        exit_code = None
        timed_out = False
        error = None

        with (
            tempfile.TemporaryFile() as stdout_file,
            tempfile.TemporaryFile() as stderr_file,
        ):
            try:
                completed = subprocess.run(
                    arguments,
                    cwd=root,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    shell=False,
                    timeout=timeout,
                    check=False,
                )
                exit_code = completed.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
            except OSError as exc:
                error = str(exc)

            stdout, stdout_truncated = self._read_output(stdout_file)
            stderr, stderr_truncated = self._read_output(stderr_file)

        return RunResult(
            command=arguments,
            cwd=root,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=perf_counter() - started,
            timed_out=timed_out,
            error=error,
            output_truncated=stdout_truncated or stderr_truncated,
        )

    def _read_output(self, stream: BinaryIO) -> tuple[str, bool]:
        stream.seek(0)
        data = stream.read(self.max_output_bytes + 1)
        text = data[: self.max_output_bytes].decode("utf-8", errors="replace")
        return text, len(data) > self.max_output_bytes
