import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from harness.agent import AgentResult, Message, ModelReply, ToolCall
import harness.cli as cli
from harness.report import RunReport
from harness.run import ReportSaveError, RunOutcome
from harness.runner import LocalRunner, RunResult


@pytest.fixture(autouse=True)
def isolated_credentials(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CLEANROOM_MODEL", raising=False)


def arguments(tmp_path, *options, command=None):
    return [
        "run",
        "--repo",
        str(tmp_path / "source"),
        "--task",
        "Fix addition",
        "--model",
        "test-model",
        "--output",
        str(tmp_path / "output"),
        *options,
        "--",
        *(command if command is not None else ["project-test"]),
    ]


def outcome(
    tmp_path,
    *,
    stop_reason="completed",
    exit_code=0,
    error=None,
    cleanup_error=None,
    interrupted=False,
    timed_out=False,
):
    agent = AgentResult(
        stop_reason=stop_reason,
        final_text="Model summary",
        iterations=1,
        tool_calls=0,
        input_tokens=3,
        output_tokens=2,
        duration_seconds=0.1,
        messages=(Message("assistant", "Model summary"),),
    )
    verification = RunResult(
        command=("project-test",),
        cwd=tmp_path,
        exit_code=exit_code,
        stdout="",
        stderr="",
        duration_seconds=0.1,
        timed_out=timed_out,
    )
    report = RunReport(
        task="Fix",
        repository=tmp_path,
        model="test-model",
        agent=agent,
        verification=verification,
        error=error,
    )
    return RunOutcome(
        report,
        tmp_path / "output",
        cleanup_error=cleanup_error,
        interrupted=interrupted,
    )


class ScriptedClient:
    def __init__(self, replies=()):
        self.replies = iter(replies)
        self.closed = False

    def complete(self, messages, tools, **options):
        return next(self.replies)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True


def stub_execution(monkeypatch, result, *, capture=None):
    client = ScriptedClient()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-key-not-logged")

    def factory(**options):
        if capture is not None:
            capture["client_options"] = options
        return client

    def run_task(config, model_client, *, output_directory):
        if capture is not None:
            capture["config"] = config
            capture["output"] = output_directory
        assert model_client is client
        return result

    monkeypatch.setattr(cli, "ClaudeClient", factory)
    monkeypatch.setattr(cli, "run_task", run_task)
    return client


@pytest.mark.parametrize("argv", [["--help"], ["run", "--help"]])
def test_help_needs_neither_credentials_nor_sdk_construction(monkeypatch, capsys, argv):
    monkeypatch.setattr(
        cli, "ClaudeClient", lambda **kwargs: pytest.fail("Help constructed a client")
    )
    with pytest.raises(SystemExit) as exc_info:
        cli.main(argv)
    assert exc_info.value.code == 0
    assert "cleanroom" in capsys.readouterr().out


def test_arguments_limits_and_literal_test_argv_reach_coordinator(
    tmp_path, monkeypatch, capsys
):
    captured = {}
    client = stub_execution(monkeypatch, outcome(tmp_path), capture=captured)
    command = [
        r"C:\Program Files\Python\python.exe",
        "-m",
        "unittest",
        "--failfast",
        "name with spaces",
    ]
    code = cli.main(
        arguments(
            tmp_path,
            "--editable",
            "calculator.py",
            "--editable",
            "config.json",
            "--test-timeout",
            "12",
            "--git-timeout",
            "8",
            "--max-iterations",
            "4",
            "--max-tool-calls",
            "7",
            "--max-output-tokens",
            "100",
            "--max-total-tokens",
            "1000",
            "--max-seconds",
            "30",
            command=command,
        )
    )
    config = captured["config"]
    assert code == 0
    assert config.test_command == tuple(command)
    assert config.editable_files == ("calculator.py", "config.json")
    assert config.test_timeout == 12
    assert config.git_timeout == 8
    assert config.agent_limits.max_iterations == 4
    assert config.agent_limits.max_tool_calls == 7
    assert config.agent_limits.max_output_tokens == 100
    assert config.agent_limits.max_total_tokens == 1000
    assert config.agent_limits.max_seconds == 30
    assert captured["client_options"]["model"] == "test-model"
    assert client.closed
    output = capsys.readouterr()
    assert "Verification: passed" in output.out
    assert "Model summary" in output.out
    assert "dummy-key-not-logged" not in output.out + output.err


def test_model_can_come_from_environment_and_explicit_option_wins(
    tmp_path, monkeypatch
):
    captured = {}
    stub_execution(monkeypatch, outcome(tmp_path), capture=captured)
    monkeypatch.setenv("CLEANROOM_MODEL", "environment-model")
    argv = arguments(tmp_path)
    position = argv.index("--model")
    del argv[position : position + 2]
    assert cli.main(argv) == 0
    assert captured["config"].model == "environment-model"
    assert cli.main(arguments(tmp_path)) == 0
    assert captured["config"].model == "test-model"


@pytest.mark.parametrize(
    "changes, expected",
    [
        ({}, 0),
        ({"exit_code": 1}, 1),
        ({"stop_reason": "iteration_limit"}, 1),
        ({"stop_reason": "refused"}, 1),
        ({"error": "Preparation failed"}, 3),
        ({"cleanup_error": "Workspace busy"}, 3),
        ({"exit_code": None}, 3),
        ({"exit_code": None, "timed_out": True}, 3),
        ({"error": "KeyboardInterrupt", "interrupted": True}, 130),
    ],
)
def test_exit_codes_distinguish_outcomes(tmp_path, monkeypatch, changes, expected):
    stub_execution(monkeypatch, outcome(tmp_path, **changes))
    assert cli.main(arguments(tmp_path)) == expected


@pytest.mark.parametrize(
    "options",
    [
        ["--max-iterations", "0"],
        ["--max-seconds", "nan"],
        ["--test-timeout", "-1"],
        ["--editable", "../outside.py"],
    ],
)
def test_invalid_configuration_fails_before_client_creation(
    tmp_path, monkeypatch, options
):
    monkeypatch.setattr(
        cli,
        "ClaudeClient",
        lambda **kwargs: pytest.fail("Invalid config created client"),
    )
    with pytest.raises(SystemExit) as exc_info:
        cli.main(arguments(tmp_path, *options))
    assert exc_info.value.code == 2


def test_missing_test_command_is_usage_error(tmp_path):
    with pytest.raises(SystemExit) as exc_info:
        cli.main(arguments(tmp_path, command=[]))
    assert exc_info.value.code == 2


def test_missing_model_is_usage_error(tmp_path):
    argv = arguments(tmp_path)
    position = argv.index("--model")
    del argv[position : position + 2]
    with pytest.raises(SystemExit) as exc_info:
        cli.main(argv)
    assert exc_info.value.code == 2


def test_missing_credentials_are_actionable(tmp_path, capsys):
    assert cli.main(arguments(tmp_path)) == 3
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_missing_optional_dependency_is_actionable(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-key")

    def missing(**kwargs):
        raise ImportError('Install Claude support with: pip install -e ".[claude]"')

    monkeypatch.setattr(cli, "ClaudeClient", missing)
    assert cli.main(arguments(tmp_path)) == 3
    assert ".[claude]" in capsys.readouterr().err


@pytest.mark.parametrize("interrupted, expected", [(False, 3), (True, 130)])
def test_report_failure_displays_retained_workspace(
    tmp_path, monkeypatch, capsys, interrupted, expected
):
    result = outcome(tmp_path)
    stub_execution(monkeypatch, result)
    retained = tmp_path / "retained-workspace"

    def failed(*args, **kwargs):
        failure = ReportSaveError(result.report, retained)
        raise failure from (
            KeyboardInterrupt() if interrupted else OSError("Disk full")
        )

    monkeypatch.setattr(cli, "run_task", failed)
    assert cli.main(arguments(tmp_path)) == expected
    assert str(retained) in capsys.readouterr().err


def test_interruption_before_coordinator_returns_has_conventional_exit(
    tmp_path, monkeypatch
):
    stub_execution(monkeypatch, outcome(tmp_path))

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(cli, "run_task", interrupted)
    assert cli.main(arguments(tmp_path)) == 130


def test_module_entry_point_help(tmp_path):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    environment.pop("ANTHROPIC_API_KEY", None)
    result = subprocess.run(
        [sys.executable, "-B", "-m", "harness", "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "cleanroom" in result.stdout


def git(root, *args):
    environment = os.environ.copy()
    environment.update({"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"})
    result = subprocess.run(
        [
            "git",
            "-c",
            "core.longpaths=true",
            "-c",
            "core.autocrlf=false",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "user.name=Cleanroom Tests",
            "-c",
            "user.email=tests@example.invalid",
            *args,
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    return result.stdout.decode("utf-8")


def test_cli_runs_coordinator_and_bundled_example_without_live_model(
    tmp_path, monkeypatch, capsys
):
    example = Path(__file__).resolve().parents[1] / "examples" / "buggy_calculator"
    source = tmp_path / "calculator-example"
    shutil.copytree(example, source)
    hooks = tmp_path / "empty-hooks"
    hooks.mkdir()
    git(source, "init", "--initial-branch=main", f"--template={hooks}")
    git(source, "config", "core.hooksPath", str(hooks))
    git(source, "add", ".")
    git(source, "commit", "-m", "Broken calculator example")
    original = (source / "calculator.py").read_bytes()
    client = ScriptedClient(
        [
            ModelReply(
                stop_reason="tool_calls",
                tool_calls=(
                    ToolCall(
                        "edit-1",
                        "replace_text",
                        {
                            "path": "calculator.py",
                            "old": "return a - b",
                            "new": "return a + b",
                        },
                    ),
                ),
            ),
            ModelReply(text="Fixed addition"),
        ]
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-key-not-logged")
    monkeypatch.setattr(cli, "ClaudeClient", lambda **kwargs: client)
    output = tmp_path / "run"
    code = cli.main(
        [
            "run",
            "--repo",
            str(source),
            "--task",
            "Fix addition",
            "--model",
            "scripted-model",
            "--output",
            str(output),
            "--editable",
            "calculator.py",
            "--",
            sys.executable,
            "-B",
            "-m",
            "unittest",
            "-v",
        ]
    )
    summary = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert code == 0
    assert summary["baseline_status"] == "failed"
    assert summary["verification_status"] == "passed"
    assert "dummy-key-not-logged" not in (output / "report.json").read_text(
        encoding="utf-8"
    )
    assert (source / "calculator.py").read_bytes() == original
    git(source, "apply", "--check", str(output / "changes.patch"))
    assert client.closed
    assert "Verification: passed" in capsys.readouterr().out


def test_bundled_example_checks_fail_before_fix_and_pass_after(tmp_path):
    example = Path(__file__).resolve().parents[1] / "examples" / "buggy_calculator"
    source = tmp_path / "example"
    shutil.copytree(example, source)
    runner = LocalRunner()
    command = [sys.executable, "-B", "-m", "unittest", "-v"]
    assert not runner.run(command, cwd=source).succeeded
    code = (source / "calculator.py").read_text(encoding="utf-8")
    (source / "calculator.py").write_text(
        code.replace("return a - b", "return a + b"), encoding="utf-8"
    )
    assert runner.run(command, cwd=source).succeeded
