"""Coordinate a task on disposable copies of a clean, local Git repository."""

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from time import perf_counter
import traceback

from .agent import AgentLimits, CodingAgent, ModelClient
from .report import RunReport, save_report
from .runner import LocalRunner, Runner
from .tools import ToolDispatcher
from .workspace import Workspace


@dataclass(frozen=True)
class RunConfig:
    repository: str | Path
    task: str
    model: str
    test_command: Sequence[str]
    editable_files: Sequence[str] = ()
    test_timeout: float = 60.0
    git_timeout: float = 30.0
    agent_limits: AgentLimits = field(default_factory=AgentLimits)

    def __post_init__(self):
        for name in ("task", "model"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be nonempty text")
        if not isinstance(self.repository, (str, Path)) or not str(self.repository):
            raise ValueError("repository must be a nonempty local path")
        object.__setattr__(self, "repository", Path(self.repository).resolve())

        if isinstance(self.test_command, (str, bytes)):
            raise ValueError("test_command must be a sequence of arguments")
        command = tuple(self.test_command)
        if (
            not command
            or any(not isinstance(argument, str) for argument in command)
            or not command[0]
            or any("\0" in argument for argument in command)
        ):
            raise ValueError("test_command must contain valid string arguments")
        object.__setattr__(self, "test_command", command)

        if isinstance(self.editable_files, (str, bytes)):
            raise ValueError("editable_files must be a sequence of relative paths")
        editable = tuple(self.editable_files)
        for path in editable:
            if not isinstance(path, str) or not path or "\0" in path:
                raise ValueError("Editable files must be nonempty relative paths")
            parts = Path(path).parts
            if (
                Path(path).is_absolute()
                or not parts
                or ".." in parts
                or any(part.casefold() == ".git" for part in parts)
            ):
                raise ValueError(
                    "Editable files must be relative paths inside the repository"
                )
        object.__setattr__(self, "editable_files", editable)

        for name in ("test_timeout", "git_timeout"):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive, finite number")
        if not isinstance(self.agent_limits, AgentLimits):
            raise ValueError("agent_limits must be an AgentLimits instance")


@dataclass(frozen=True)
class RunOutcome:
    report: RunReport
    output_directory: Path
    cleanup_error: str | None = None
    retained_workspace: Path | None = None
    interrupted: bool = False


class RepositoryError(RuntimeError):
    """Repository preparation or patch extraction could not complete."""


class ReportSaveError(RuntimeError):
    """Saving failed; the in-memory report and scratch directory remain available."""

    def __init__(self, report: RunReport, workspace: Path | None):
        self.report = report
        self.workspace = workspace
        super().__init__(f"Could not save the report; retained workspace: {workspace}")


class _Git:
    def __init__(self, hooks: Path, timeout: float):
        self.hooks = hooks
        self.timeout = timeout
        self.environment = os.environ.copy()
        for key in list(self.environment):
            if key in {
                "GIT_DIR",
                "GIT_WORK_TREE",
                "GIT_INDEX_FILE",
                "GIT_COMMON_DIR",
                "GIT_OBJECT_DIRECTORY",
                "GIT_ALTERNATE_OBJECT_DIRECTORIES",
                "GIT_CONFIG",
                "GIT_CONFIG_COUNT",
                "GIT_CONFIG_PARAMETERS",
                "GIT_EXTERNAL_DIFF",
            } or key.startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_")):
                self.environment.pop(key)
        self.environment.update(
            {
                "GIT_OPTIONAL_LOCKS": "0",
                "GIT_TERMINAL_PROMPT": "0",
                "GIT_ALLOW_PROTOCOL": "file",
                "GIT_LFS_SKIP_SMUDGE": "1",
            }
        )
        # Read-only source checks respect the user's line-ending configuration.
        # Disposable Git operations use only their explicitly captured settings.
        self.source_environment = self.environment.copy()
        self.environment.update(
            {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
        )

    def run(
        self, cwd: Path, *arguments: str, source_configuration: bool = False
    ) -> str:
        completed = subprocess.run(
            [
                "git",
                "--no-optional-locks",
                "--literal-pathspecs",
                "-c",
                "core.fsmonitor=false",
                "-c",
                f"core.hooksPath={self.hooks}",
                "-c",
                "core.longpaths=true",
                *arguments,
            ],
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            env=self.source_environment if source_configuration else self.environment,
            timeout=self.timeout,
            check=False,
        )
        if completed.returncode != 0:
            detail = (
                (completed.stderr or completed.stdout)
                .decode("utf-8", errors="replace")
                .strip()
            )
            raise RepositoryError(detail or f"Git exited with {completed.returncode}")
        # Do not use text-mode newline conversion: CRLF in patch context must
        # survive byte-for-byte so the saved patch remains applicable.
        return completed.stdout.decode("utf-8")


def _prepare_snapshot(
    source: Path, snapshot: Path, git: _Git
) -> tuple[str, set[str], dict[str, str]]:
    if not source.is_dir():
        raise RepositoryError(f"Repository directory does not exist: {source}")
    top = Path(git.run(source, "rev-parse", "--show-toplevel").strip()).resolve()
    if top != source:
        raise RepositoryError("repository must name the Git working-tree root")
    if git.run(
        source,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        source_configuration=True,
    ).strip():
        raise RepositoryError("Source repository must have a clean working tree")
    line_endings = {
        name: git.run(
            source,
            "config",
            "--default",
            default,
            "--get",
            name,
            source_configuration=True,
        ).strip()
        for name, default in (("core.autocrlf", "false"), ("core.eol", "native"))
    }
    revision = git.run(source, "rev-parse", "--verify", "HEAD^{commit}").strip()
    tracked = set()
    for entry in git.run(source, "ls-tree", "-r", "-z", revision).split("\0"):
        if not entry:
            continue
        metadata, path = entry.split("\t", 1)
        if metadata.split(" ", 1)[0] not in ("100644", "100755"):
            raise RepositoryError(
                "This version supports regular tracked files, not symlinks or submodules"
            )
        tracked.add(path)

    git.run(
        source,
        "clone",
        "--local",
        "--no-hardlinks",
        "--dissociate",
        "--no-checkout",
        f"--template={git.hooks}",
        "--config",
        f"core.hooksPath={git.hooks}",
        "--config",
        "core.fsmonitor=false",
        "--config",
        f"core.autocrlf={line_endings['core.autocrlf']}",
        "--config",
        f"core.eol={line_endings['core.eol']}",
        "--config",
        "core.longpaths=true",
        "--",
        str(source),
        str(snapshot),
    )
    git.run(snapshot, "checkout", "--detach", revision)
    return revision, tracked, line_endings


def _approved_files(
    config: RunConfig, snapshot: Path, tracked: set[str]
) -> tuple[str, ...]:
    approved = set()
    for path in config.editable_files:
        target = (snapshot / path).resolve()
        if not target.is_relative_to(snapshot) or not target.is_file():
            raise RepositoryError(
                f"Editable file is not a regular tracked file: {path}"
            )
        relative = target.relative_to(snapshot).as_posix()
        if relative not in tracked:
            raise RepositoryError(f"Editable file is not tracked: {path}")
        approved.add(relative)
    return tuple(sorted(approved))


def _copy_approved(
    workspace: Path, verification: Path, approved: Sequence[str]
) -> None:
    for path in approved:
        source = workspace / path
        # Reject newly introduced links, including links through a parent. This
        # is a conservative transfer policy, not complete host isolation.
        if source.resolve() != source or not source.is_file():
            raise RepositoryError(f"Approved file was removed or redirected: {path}")
        shutil.copy2(source, verification / path)


def _cleanup_workspace(directory: Path, parent: Path) -> None:
    if (
        directory.resolve() != directory
        or directory.parent.resolve() != parent
        or not directory.is_relative_to(parent)
        or not directory.name.startswith("cleanroom-")
    ):
        raise RepositoryError("Refusing to clean an unexpected workspace location")

    def retry_readonly(function, path, error):
        target = Path(path).resolve()
        if not isinstance(error, PermissionError) or not target.is_relative_to(
            directory
        ):
            raise error
        target.chmod(0o700 if target.is_dir() else 0o600)
        function(path)

    if sys.version_info >= (3, 12):
        shutil.rmtree(directory, onexc=retry_readonly)
    else:
        shutil.rmtree(
            directory, onerror=lambda fn, path, info: retry_readonly(fn, path, info[1])
        )


def run_task(
    config: RunConfig,
    client: ModelClient,
    *,
    output_directory: str | Path,
    runner: Runner | None = None,
) -> RunOutcome:
    """Prepare, baseline, edit, independently verify, save, then clean up.

    The source must be a clean local Git root. Only committed regular files are
    copied. Verification gets a fresh copy plus approved file edits. The runner
    still executes with its own permissions; this is not a host security boundary.
    """
    output = Path(output_directory).resolve()
    if output.is_relative_to(config.repository):
        raise ValueError("Report output must be outside the source repository")
    if output.exists():
        raise FileExistsError(f"Report output already exists: {output}")
    selected_runner = runner if runner is not None else LocalRunner()
    started = perf_counter()
    scratch = None
    scratch_parent = Path(tempfile.gettempdir()).resolve()
    snapshot = work = None
    git = None
    revision = None
    baseline = agent = verification = None
    approved = ()
    patch = ""
    error = None
    failure_traceback = None
    interrupted = False
    unapproved = []
    line_endings = {}

    try:
        scratch = Path(
            tempfile.mkdtemp(prefix="cleanroom-", dir=scratch_parent)
        ).resolve()
        hooks = scratch / "empty-hooks"
        hooks.mkdir()
        git = _Git(hooks, config.git_timeout)
        snapshot = scratch / "snapshot"
        revision, tracked, line_endings = _prepare_snapshot(
            config.repository, snapshot, git
        )
        approved = _approved_files(config, snapshot, tracked)

        baseline_directory = scratch / "baseline"
        shutil.copytree(snapshot, baseline_directory)
        baseline = selected_runner.run(
            config.test_command, cwd=baseline_directory, timeout=config.test_timeout
        )
        if (
            baseline.error is not None
            or baseline.timed_out
            or baseline.exit_code is None
        ):
            raise RepositoryError(
                "Baseline command could not complete; agent was not started"
            )

        work = scratch / "agent"
        shutil.copytree(snapshot, work)
        dispatcher = ToolDispatcher(
            Workspace(work),
            selected_runner,
            test_command=config.test_command,
            test_timeout=config.test_timeout,
            editable_files=approved,
        )
        try:
            agent = CodingAgent(client, dispatcher, limits=config.agent_limits).run(
                config.task
            )
            if agent.error is not None:
                error = f"Agent stopped ({agent.stop_reason}): {agent.error}"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            failure_traceback = traceback.format_exc()

        changed = git.run(
            snapshot,
            f"--work-tree={work}",
            "diff",
            "--name-only",
            "-z",
            "--no-ext-diff",
            "--no-textconv",
            revision,
        ).split("\0")
        unapproved = sorted(path for path in changed if path and path not in approved)
        if unapproved:
            detail = "Unapproved tracked changes were excluded: " + ", ".join(
                unapproved
            )
            error = f"{error}; {detail}" if error else detail

        verification_directory = scratch / "verification"
        shutil.copytree(snapshot, verification_directory)
        _copy_approved(work, verification_directory, approved)
        verification = selected_runner.run(
            config.test_command, cwd=verification_directory, timeout=config.test_timeout
        )
    except (Exception, KeyboardInterrupt) as exc:
        interrupted = isinstance(exc, KeyboardInterrupt)
        detail = f"{type(exc).__name__}: {exc}"
        error = f"{error}; {detail}" if error else detail
        failure_traceback = traceback.format_exc()

    # Use the pristine index, even if locally executed code altered the agent's
    # Git metadata. Extract the approved patch on both successful and failed runs.
    if work is not None and work.is_dir() and snapshot is not None and approved:
        try:
            patch = git.run(
                snapshot,
                f"--work-tree={work}",
                "diff",
                "--binary",
                "--no-ext-diff",
                "--no-textconv",
                revision,
                "--",
                *approved,
            )
        except Exception as exc:
            detail = f"Patch capture failed: {type(exc).__name__}: {exc}"
            error = f"{error}; {detail}" if error else detail

    configuration = {
        **asdict(config),
        "runner": type(selected_runner).__name__,
        "editable_files": list(approved),
        "unapproved_changes": unapproved,
        "git_line_endings": line_endings,
        "interrupted": interrupted,
    }
    if failure_traceback is not None:
        configuration["failure_traceback"] = failure_traceback
    report = RunReport(
        task=config.task,
        repository=config.repository,
        model=config.model,
        agent=agent,
        baseline=baseline,
        verification=verification,
        source_revision=revision,
        patch=patch,
        error=error,
        duration_seconds=perf_counter() - started,
        configuration=configuration,
    )
    try:
        saved = save_report(report, output)
    except (Exception, KeyboardInterrupt) as exc:
        # Retain the only editable copy if artifact persistence failed.
        raise ReportSaveError(report, scratch) from exc

    cleanup_error = None
    if scratch is not None:
        try:
            _cleanup_workspace(scratch, scratch_parent)
        except Exception as exc:
            cleanup_error = (
                f"{type(exc).__name__}: {exc}; retained workspace: {scratch}"
            )
            try:
                with (saved / "cleanup_error.txt").open(
                    "x", encoding="utf-8"
                ) as stream:
                    stream.write(cleanup_error + "\n")
            except OSError as warning_error:
                cleanup_error += f"; could not save cleanup warning: {warning_error}"

    return RunOutcome(
        report=report,
        output_directory=saved,
        cleanup_error=cleanup_error,
        retained_workspace=scratch if cleanup_error else None,
        interrupted=interrupted,
    )
