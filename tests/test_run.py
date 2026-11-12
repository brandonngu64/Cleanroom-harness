from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from harness.agent import AgentLimits, ModelError, ModelReply, ToolCall
import harness.run as run_module
from harness.run import ReportSaveError, RunConfig, run_task
from harness.runner import LocalRunner


def git(root, *arguments):
    environment = os.environ.copy()
    environment.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_COUNT": "0",
        }
    )
    completed = subprocess.run(
        [
            "git",
            "-c",
            "core.autocrlf=false",
            "-c",
            "core.longpaths=true",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "user.name=Cleanroom Tests",
            "-c",
            "user.email=tests@example.invalid",
            *arguments,
        ],
        cwd=root,
        capture_output=True,
        env=environment,
        check=False,
    )
    if completed.returncode:
        raise AssertionError(completed.stderr.decode("utf-8", errors="replace"))
    return completed.stdout.decode("utf-8")


class ScriptedClient:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    def complete(self, messages, tools, **options):
        self.requests.append(messages)
        reply = next(self.replies)
        if isinstance(reply, BaseException):
            raise reply
        return reply() if callable(reply) else reply


class RecordingRunner:
    def __init__(self):
        self.calls = []
        self.delegate = LocalRunner()

    def run(self, command, *, cwd, timeout=60):
        self.calls.append(Path(cwd))
        return self.delegate.run(command, cwd=cwd, timeout=timeout)


def edit_reply():
    return ModelReply(
        stop_reason="tool_calls",
        tool_calls=(
            ToolCall(
                "edit-1",
                "replace_text",
                {"path": "calculator.py", "old": "return a - b", "new": "return a + b"},
            ),
        ),
        input_tokens=3,
        output_tokens=2,
    )


@pytest.fixture
def repository(tmp_path, monkeypatch):
    # Keep all coordinator scratch directories inside this test's temporary root.
    monkeypatch.setattr(run_module.tempfile, "gettempdir", lambda: str(tmp_path))
    source = tmp_path / "source"
    source.mkdir()
    template = tmp_path / "empty-template"
    template.mkdir()
    git(source, "init", "--initial-branch=main", f"--template={template}")
    git(source, "config", "core.hooksPath", str(template))
    (source / "calculator.py").write_text(
        "def add(a, b):\n    return a - b\n", encoding="utf-8"
    )
    (source / "check.py").write_text(
        "from calculator import add\nassert add(2, 3) == 5\n", encoding="utf-8"
    )
    (source / ".gitignore").write_text("__pycache__/\nprivate.txt\n", encoding="utf-8")
    git(source, "add", "--", "calculator.py", "check.py", ".gitignore")
    git(source, "commit", "-m", "Broken calculator fixture")
    return source


@pytest.fixture
def config(repository):
    return RunConfig(
        repository=repository,
        task="Fix addition",
        model="scripted-model",
        test_command=[sys.executable, "-B", "check.py"],
        editable_files=["calculator.py"],
    )


def test_complete_run_saves_applicable_patch_before_cleanup(
    config, repository, tmp_path, monkeypatch
):
    original = (repository / "calculator.py").read_bytes()
    revision = git(repository, "rev-parse", "HEAD").strip()
    output = tmp_path / "artifacts"
    client = ScriptedClient(
        [
            edit_reply(),
            ModelReply(
                stop_reason="tool_calls",
                tool_calls=(ToolCall("tests-1", "run_tests", {}),),
            ),
            ModelReply(text="Fixed"),
        ]
    )
    runner = RecordingRunner()
    actual_cleanup = run_module._cleanup_workspace

    def checked_cleanup(directory, parent):
        assert (output / "report.json").exists()
        assert "return a + b" in (output / "changes.patch").read_text(encoding="utf-8")
        actual_cleanup(directory, parent)

    monkeypatch.setattr(run_module, "_cleanup_workspace", checked_cleanup)

    outcome = run_task(config, client, output_directory=output, runner=runner)

    assert outcome.report.error is None
    assert outcome.report.baseline.succeeded is False
    assert outcome.report.verification.succeeded is True
    assert outcome.report.agent.stop_reason == "completed"
    assert outcome.report.source_revision == revision
    assert outcome.cleanup_error is None
    assert outcome.retained_workspace is None
    assert len(runner.calls) == 3
    assert len(set(runner.calls)) == 3
    assert repository not in runner.calls
    assert all(not path.exists() for path in runner.calls)
    assert (repository / "calculator.py").read_bytes() == original
    assert git(repository, "status", "--porcelain") == ""
    assert git(repository, "rev-parse", "HEAD").strip() == revision
    git(repository, "apply", "--check", str(output / "changes.patch"))
    summary = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert summary["verification_status"] == "passed"
    assert not list(tmp_path.glob("cleanroom-*"))


def test_final_verification_runs_even_without_a_model_test_request(config, tmp_path):
    runner = RecordingRunner()
    outcome = run_task(
        config,
        ScriptedClient([edit_reply(), ModelReply(text="Done")]),
        output_directory=tmp_path / "run",
        runner=runner,
    )
    assert len(runner.calls) == 2
    assert outcome.report.verification.succeeded


@pytest.mark.parametrize("scope", ["local", "global"])
def test_autocrlf_settings_are_preserved(
    config, repository, tmp_path, monkeypatch, scope
):
    if scope == "local":
        git(repository, "config", "core.autocrlf", "true")
    else:
        configuration = tmp_path / "global-git-config"
        configuration.write_text("[core]\n\tautocrlf = true\n", encoding="utf-8")
        monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(configuration))
    git(repository, "-c", "core.autocrlf=true", "add", "--renormalize", ".")
    git(repository, "commit", "--allow-empty", "-m", "Normalize line endings")
    assert git(repository, "-c", "core.autocrlf=true", "status", "--porcelain") == ""
    original = (repository / "calculator.py").read_bytes()
    outcome = run_task(
        config,
        ScriptedClient([edit_reply(), ModelReply(text="Done")]),
        output_directory=tmp_path / "run",
    )
    assert outcome.report.error is None
    assert outcome.report.verification.succeeded
    assert outcome.report.configuration["git_line_endings"]["core.autocrlf"] == "true"
    assert (repository / "calculator.py").read_bytes() == original
    git(
        repository,
        "-c",
        "core.autocrlf=true",
        "apply",
        "--check",
        str(outcome.output_directory / "changes.patch"),
    )


def test_completion_alone_does_not_make_verification_pass(config, tmp_path):
    outcome = run_task(
        config,
        ScriptedClient([ModelReply(text="All fixed")]),
        output_directory=tmp_path / "run",
    )
    assert outcome.report.agent.stop_reason == "completed"
    assert not outcome.report.verification.succeeded
    assert outcome.report.patch == ""


def test_verification_uses_original_tests_after_workspace_tampering(
    config, repository, tmp_path
):
    original_check = (repository / "check.py").read_bytes()

    def tampered_reply():
        work = next(tmp_path.glob("cleanroom-*")) / "agent"
        (work / "check.py").write_text("print('fake pass')\n", encoding="utf-8")
        (work / "calculator.py").write_text(
            "def add(a, b):\n    return 42\n", encoding="utf-8"
        )
        return ModelReply(text="Claimed success")

    outcome = run_task(
        config, ScriptedClient([tampered_reply]), output_directory=tmp_path / "run"
    )

    assert not outcome.report.verification.succeeded
    assert "Unapproved tracked changes" in outcome.report.error
    assert outcome.report.configuration["unapproved_changes"] == ["check.py"]
    assert "check.py" not in outcome.report.patch
    assert "return 42" in outcome.report.patch
    assert (repository / "check.py").read_bytes() == original_check


def test_baseline_side_effects_do_not_enter_the_agent_copy(config, tmp_path):
    class MutatingBaselineRunner(RecordingRunner):
        def run(self, command, *, cwd, timeout=60):
            if not self.calls:
                (Path(cwd) / "calculator.py").write_text(
                    "def add(a, b):\n    return 999\n", encoding="utf-8"
                )
            return super().run(command, cwd=cwd, timeout=timeout)

    def inspect():
        work = next(tmp_path.glob("cleanroom-*")) / "agent"
        assert "return a - b" in (work / "calculator.py").read_text(encoding="utf-8")
        return edit_reply()

    outcome = run_task(
        config,
        ScriptedClient([inspect, ModelReply(text="Done")]),
        output_directory=tmp_path / "run",
        runner=MutatingBaselineRunner(),
    )
    assert outcome.report.error is None
    assert outcome.report.verification.succeeded


@pytest.mark.parametrize("kind", ["tracked", "staged", "untracked"])
def test_dirty_source_is_reported_without_calling_the_model(
    config, repository, tmp_path, kind
):
    if kind == "untracked":
        (repository / "untracked.txt").write_text("not committed", encoding="utf-8")
    else:
        (repository / "calculator.py").write_text("changed", encoding="utf-8")
        if kind == "staged":
            git(repository, "add", "calculator.py")
    client = ScriptedClient([])
    outcome = run_task(config, client, output_directory=tmp_path / "run")
    assert "clean working tree" in outcome.report.error
    assert outcome.report.agent is None
    assert outcome.report.baseline is None
    assert client.requests == []
    assert (outcome.output_directory / "report.json").exists()
    assert not list(tmp_path.glob("cleanroom-*"))


def test_ignored_source_files_are_not_copied(config, repository, tmp_path):
    (repository / "private.txt").write_text("local-only data", encoding="utf-8")

    def inspect():
        work = next(tmp_path.glob("cleanroom-*")) / "agent"
        assert not (work / "private.txt").exists()
        return ModelReply(text="Inspected")

    outcome = run_task(
        config, ScriptedClient([inspect]), output_directory=tmp_path / "run"
    )
    assert outcome.report.error is None
    assert (repository / "private.txt").read_text(encoding="utf-8") == "local-only data"


@pytest.mark.parametrize("missing", [False, True])
def test_invalid_repository_preparation_is_saved(config, tmp_path, missing):
    source = tmp_path / "invalid"
    if not missing:
        source.mkdir()
    client = ScriptedClient([])
    outcome = run_task(
        replace(config, repository=source), client, output_directory=tmp_path / "run"
    )
    assert outcome.report.error
    assert outcome.report.agent is None
    assert client.requests == []


def test_subdirectory_is_not_silently_treated_as_repository_root(
    config, repository, tmp_path
):
    directory = repository / "subdirectory"
    directory.mkdir()
    outcome = run_task(
        replace(config, repository=directory),
        ScriptedClient([]),
        output_directory=tmp_path / "run",
    )
    assert "working-tree root" in outcome.report.error


def test_nonexistent_editable_file_is_reported_before_baseline(config, tmp_path):
    outcome = run_task(
        replace(config, editable_files=["missing.py"]),
        ScriptedClient([]),
        output_directory=tmp_path / "run",
    )
    assert "Editable file" in outcome.report.error
    assert outcome.report.baseline is None


@pytest.mark.parametrize("kind", ["startup", "timeout"])
def test_baseline_execution_failure_stops_before_model_requests(config, tmp_path, kind):
    if kind == "startup":
        command = [str(tmp_path / "missing-program")]
    else:
        command = [sys.executable, "-c", "import time; time.sleep(10)"]
    client = ScriptedClient([])
    outcome = run_task(
        replace(config, test_command=command, test_timeout=0.1),
        client,
        output_directory=tmp_path / "run",
    )
    assert "Baseline command could not complete" in outcome.report.error
    assert outcome.report.baseline is not None
    assert outcome.report.verification is None
    assert client.requests == []


def test_model_failure_still_gets_final_verification(config, tmp_path):
    runner = RecordingRunner()
    outcome = run_task(
        config,
        ScriptedClient([ModelError("Provider unavailable")]),
        output_directory=tmp_path / "run",
        runner=runner,
    )
    assert outcome.report.agent.stop_reason == "model_error"
    assert "Provider unavailable" in outcome.report.error
    assert outcome.report.verification is not None
    assert len(runner.calls) == 2


def test_partial_edits_and_traceback_survive_unexpected_model_failure(config, tmp_path):
    outcome = run_task(
        config,
        ScriptedClient([edit_reply(), RuntimeError("Client bug")]),
        output_directory=tmp_path / "run",
    )
    assert "Client bug" in outcome.report.error
    assert "RuntimeError" in outcome.report.configuration["failure_traceback"]
    assert "return a + b" in outcome.report.patch
    assert outcome.report.verification.succeeded


def test_agent_limit_does_not_skip_final_verification(config, tmp_path):
    outcome = run_task(
        replace(config, agent_limits=AgentLimits(max_iterations=1)),
        ScriptedClient([edit_reply()]),
        output_directory=tmp_path / "run",
    )
    assert outcome.report.agent.stop_reason == "iteration_limit"
    assert outcome.report.verification.succeeded


def test_interruption_saves_partial_patch_and_cleans_up(config, tmp_path):
    client = ScriptedClient([edit_reply(), KeyboardInterrupt()])
    outcome = run_task(config, client, output_directory=tmp_path / "run")
    assert "KeyboardInterrupt" in outcome.report.error
    assert outcome.interrupted
    assert outcome.report.configuration["interrupted"] is True
    assert "return a + b" in outcome.report.patch
    assert outcome.report.verification is None
    assert (outcome.output_directory / "report.json").exists()
    assert not list(tmp_path.glob("cleanroom-*"))


def test_symlink_entries_are_explicitly_unsupported(config, repository, tmp_path):
    # Represent a symlink in Git's index without requiring Windows symlink rights.
    git(repository, "config", "core.symlinks", "false")
    (repository / "link").write_text("calculator.py", encoding="utf-8")
    git(repository, "add", "link")
    object_id = git(repository, "rev-parse", ":link").strip()
    git(repository, "update-index", "--cacheinfo", f"120000,{object_id},link")
    git(repository, "commit", "-m", "Add a symlink entry")
    assert git(repository, "status", "--porcelain") == ""
    client = ScriptedClient([])
    outcome = run_task(config, client, output_directory=tmp_path / "run")
    assert "symlinks or submodules" in outcome.report.error
    assert client.requests == []


def test_existing_output_is_rejected_before_any_work(config, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_task(config, ScriptedClient([]), output_directory=output)
    assert marker.read_text(encoding="utf-8") == "keep"
    assert not list(tmp_path.glob("cleanroom-*"))


def test_output_inside_source_is_rejected(config, repository):
    output = repository / "runs" / "run-1"
    with pytest.raises(ValueError, match="outside"):
        run_task(config, ScriptedClient([]), output_directory=output)
    assert not output.exists()


def test_report_save_failure_retains_workspace_and_in_memory_report(
    config, tmp_path, monkeypatch
):
    def failing_save(report, output):
        raise OSError("Disk failure")

    monkeypatch.setattr(run_module, "save_report", failing_save)
    with pytest.raises(ReportSaveError) as exc_info:
        run_task(
            config,
            ScriptedClient([edit_reply(), ModelReply(text="Done")]),
            output_directory=tmp_path / "run",
        )
    failure = exc_info.value
    assert isinstance(failure.__cause__, OSError)
    assert failure.workspace.exists()
    assert "return a + b" in failure.report.patch
    assert "return a + b" in (failure.workspace / "agent" / "calculator.py").read_text(
        encoding="utf-8"
    )
    run_module._cleanup_workspace(failure.workspace, tmp_path)


def test_interruption_during_report_saving_retains_workspace(
    config, tmp_path, monkeypatch
):
    def interrupted_save(report, output):
        raise KeyboardInterrupt()

    monkeypatch.setattr(run_module, "save_report", interrupted_save)
    with pytest.raises(ReportSaveError) as exc_info:
        run_task(
            config,
            ScriptedClient([edit_reply(), ModelReply(text="Done")]),
            output_directory=tmp_path / "run",
        )
    assert isinstance(exc_info.value.__cause__, KeyboardInterrupt)
    assert exc_info.value.workspace.exists()
    assert "return a + b" in exc_info.value.report.patch
    run_module._cleanup_workspace(exc_info.value.workspace, tmp_path)


def test_cleanup_failure_is_returned_and_recorded(config, tmp_path, monkeypatch):
    actual_cleanup = run_module._cleanup_workspace

    def failing_cleanup(directory, parent):
        raise PermissionError("Workspace busy")

    monkeypatch.setattr(run_module, "_cleanup_workspace", failing_cleanup)
    outcome = run_task(
        config,
        ScriptedClient([ModelReply(text="Done")]),
        output_directory=tmp_path / "run",
    )
    assert "Workspace busy" in outcome.cleanup_error
    assert outcome.retained_workspace.exists()
    assert "Workspace busy" in (
        outcome.output_directory / "cleanup_error.txt"
    ).read_text(encoding="utf-8")
    actual_cleanup(outcome.retained_workspace, tmp_path)


def test_cleanup_refuses_a_directory_it_does_not_own(repository, tmp_path):
    with pytest.raises(run_module.RepositoryError, match="unexpected"):
        run_module._cleanup_workspace(repository, tmp_path)
    assert repository.exists()


def test_mutable_configuration_sequences_are_copied(repository):
    command = [sys.executable, "-B", "check.py"]
    editable = ["calculator.py"]
    config = RunConfig(
        repository=repository,
        task="Fix",
        model="test",
        test_command=command,
        editable_files=editable,
    )
    command.clear()
    editable.clear()
    assert config.test_command
    assert config.editable_files == ("calculator.py",)


@pytest.mark.parametrize(
    "changes",
    [
        {"task": ""},
        {"model": ""},
        {"repository": ""},
        {"test_command": []},
        {"test_command": "python check.py"},
        {"test_command": [123]},
        {"editable_files": "calculator.py"},
        {"editable_files": ["../outside"]},
        {"editable_files": [".git/config"]},
        {"test_timeout": 0},
        {"git_timeout": float("nan")},
        {"agent_limits": None},
    ],
)
def test_invalid_run_configuration_is_rejected(config, changes):
    with pytest.raises(ValueError):
        replace(config, **changes)
