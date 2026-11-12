from dataclasses import asdict
import json
from pathlib import Path
import sys

import pytest

from harness.runner import LocalRunner, RunResult
from harness.tools import ToolDispatcher, get_tool_definitions
from harness.workspace import Workspace


class RecordingRunner:
    """An alternate runner that records requests without launching commands."""

    def __init__(self):
        self.calls = []
        self.exit_code = 0
        self.timed_out = False
        self.error = None

    def run(self, command, *, cwd, timeout=60.0):
        self.calls.append((tuple(command), Path(cwd), timeout))
        return RunResult(
            command=tuple(command),
            cwd=Path(cwd),
            exit_code=self.exit_code,
            stdout="test output",
            stderr="",
            duration_seconds=0.1,
            timed_out=self.timed_out,
            error=self.error,
        )


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "calculator.py").write_text(
        "def add(a, b):\n    return a - b\n", encoding="utf-8"
    )
    tests = root / "tests"
    tests.mkdir()
    (tests / "test_calculator.py").write_text("trusted tests\n", encoding="utf-8")
    git = root / ".git"
    git.mkdir()
    (git / "config").write_text("git metadata\n", encoding="utf-8")
    return Workspace(root)


@pytest.fixture
def runner():
    return RecordingRunner()


@pytest.fixture
def dispatcher(workspace, runner):
    return ToolDispatcher(
        workspace,
        runner,
        test_command=["project-test", "--check"],
        test_timeout=12,
        editable_files=["calculator.py"],
    )


def test_definitions_are_json_serializable_and_describe_closed_schemas():
    definitions = get_tool_definitions()
    json.dumps(definitions)
    assert {definition["name"] for definition in definitions} == {
        "list_files",
        "read_file",
        "replace_text",
        "run_tests",
    }
    for definition in definitions:
        schema = definition["parameters"]
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])
        assert all(value["type"] == "string" for value in schema["properties"].values())
    test_tool = next(item for item in definitions if item["name"] == "run_tests")
    assert test_tool["parameters"]["properties"] == {}


def test_definitions_are_fresh_on_each_call():
    definitions = get_tool_definitions()
    definitions[0]["parameters"]["properties"]["injected"] = {"type": "string"}
    assert get_tool_definitions()[0]["parameters"]["properties"] == {}


def test_lists_workspace_files(dispatcher):
    result = dispatcher.dispatch("list_files", {})
    assert not result.is_error
    assert result.data["files"] == sorted(
        ["calculator.py", str(Path("tests") / "test_calculator.py")]
    )


def test_reads_file(dispatcher):
    result = dispatcher.dispatch("read_file", {"path": "calculator.py"})
    assert not result.is_error
    assert result.data["content"] == "def add(a, b):\n    return a - b\n"


def test_replaces_text_in_an_approved_file(dispatcher, workspace):
    result = dispatcher.dispatch(
        "replace_text",
        {"path": "calculator.py", "old": "return a - b", "new": "return a + b"},
    )
    assert not result.is_error
    assert result.data["updated"] is True
    assert workspace.read_file("calculator.py") == "def add(a, b):\n    return a + b\n"


def test_canonical_path_alias_can_edit_the_same_approved_file(dispatcher, workspace):
    result = dispatcher.dispatch(
        "replace_text",
        {
            "path": "unused/../calculator.py",
            "old": "return a - b",
            "new": "return a + b",
        },
    )
    assert not result.is_error
    assert "return a + b" in workspace.read_file("calculator.py")


@pytest.mark.parametrize("old", ["", "missing", "a"])
def test_invalid_replacement_preserves_source(dispatcher, workspace, old):
    original = workspace.read_file("calculator.py")
    result = dispatcher.dispatch(
        "replace_text", {"path": "calculator.py", "old": old, "new": "changed"}
    )
    assert result.is_error
    assert workspace.read_file("calculator.py") == original


def test_unapproved_test_file_cannot_be_edited(dispatcher, workspace):
    result = dispatcher.dispatch(
        "replace_text",
        {"path": "tests/test_calculator.py", "old": "trusted", "new": "fake"},
    )
    assert result.is_error
    assert workspace.read_file("tests/test_calculator.py") == "trusted tests\n"


def test_edits_are_disabled_by_default(workspace, runner):
    dispatcher = ToolDispatcher(workspace, runner, test_command=["project-test"])
    original = workspace.read_file("calculator.py")
    result = dispatcher.dispatch(
        "replace_text",
        {"path": "calculator.py", "old": "return a - b", "new": "return a + b"},
    )
    assert result.is_error
    assert workspace.read_file("calculator.py") == original


def test_unknown_tool_does_not_run_a_command(dispatcher, runner):
    result = dispatcher.dispatch("shell", {"command": "anything"})
    assert result.is_error
    assert "Unknown tool" in result.error
    assert runner.calls == []


@pytest.mark.parametrize("name", [None, [], 123])
def test_invalid_tool_name_returns_an_error(dispatcher, runner, name):
    result = dispatcher.dispatch(name, {})
    assert result.is_error
    assert runner.calls == []


@pytest.mark.parametrize(
    "name, arguments",
    [
        ("read_file", {}),
        ("read_file", {"path": 123}),
        ("read_file", {"path": None}),
        ("read_file", None),
        ("read_file", []),
        ("read_file", "{}"),
        ("read_file", {1: "calculator.py"}),
        ("replace_text", {"path": "calculator.py", "old": "a"}),
        ("list_files", {"path": "."}),
        ("run_tests", {"command": ["different-program"]}),
        ("run_tests", {"timeout": 999}),
        ("run_tests", {"cwd": ".."}),
        ("run_tests", {"runner": "different-backend"}),
    ],
)
def test_invalid_arguments_are_rejected_before_execution(
    dispatcher, runner, name, arguments
):
    result = dispatcher.dispatch(name, arguments)
    assert result.is_error
    assert result.data is None
    assert runner.calls == []


@pytest.mark.parametrize("path", ["missing.txt", ".", "../outside.txt", ".git/config"])
def test_workspace_errors_become_tool_errors(dispatcher, path):
    result = dispatcher.dispatch("read_file", {"path": path})
    assert result.is_error
    assert result.error


def test_runs_only_the_configured_command(dispatcher, runner, workspace):
    result = dispatcher.dispatch("run_tests", {})
    assert runner.calls == [(("project-test", "--check"), workspace.root, 12)]
    assert not result.is_error
    assert result.data["succeeded"]
    assert result.data["command"] == ["project-test", "--check"]
    assert result.data["cwd"] == str(workspace.root)
    json.dumps(asdict(result))


@pytest.mark.parametrize(
    "exit_code, timed_out, error, is_error",
    [
        (1, False, None, False),
        (None, True, None, True),
        (None, False, "Program could not start", True),
    ],
)
def test_test_failures_differ_from_execution_errors(
    dispatcher, runner, exit_code, timed_out, error, is_error
):
    runner.exit_code = exit_code
    runner.timed_out = timed_out
    runner.error = error
    result = dispatcher.dispatch("run_tests", {})
    assert result.is_error is is_error
    assert not result.data["succeeded"]
    assert result.data["exit_code"] == exit_code
    assert result.data["timed_out"] is timed_out


def test_command_configuration_is_copied(workspace, runner):
    command = ["project-test"]
    editable_files = ["calculator.py"]
    dispatcher = ToolDispatcher(
        workspace, runner, test_command=command, editable_files=editable_files
    )
    command[0] = "changed-program"
    editable_files.append("tests/test_calculator.py")
    dispatcher.dispatch("run_tests", {})
    assert runner.calls[0][0] == ("project-test",)
    result = dispatcher.dispatch(
        "replace_text",
        {"path": "tests/test_calculator.py", "old": "trusted", "new": "fake"},
    )
    assert result.is_error


def test_integrates_with_local_runner(workspace):
    dispatcher = ToolDispatcher(
        workspace,
        LocalRunner(),
        test_command=[sys.executable, "-I", "-B", "-c", "print('command ran')"],
    )
    result = dispatcher.dispatch("run_tests", {})
    assert not result.is_error
    assert result.data["succeeded"]
    assert result.data["stdout"].strip() == "command ran"


def test_unexpected_programming_errors_propagate(dispatcher, runner, monkeypatch):
    def broken_run(*args, **kwargs):
        raise RuntimeError("Internal bug")

    monkeypatch.setattr(runner, "run", broken_run)
    with pytest.raises(RuntimeError, match="Internal bug"):
        dispatcher.dispatch("run_tests", {})


@pytest.mark.parametrize("command", [[], "project-test", [""], [123], ["a\0b"]])
def test_invalid_command_configuration(workspace, runner, command):
    with pytest.raises(ValueError):
        ToolDispatcher(workspace, runner, test_command=command)


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), True])
def test_invalid_timeout_configuration(workspace, runner, timeout):
    with pytest.raises(ValueError):
        ToolDispatcher(
            workspace, runner, test_command=["project-test"], test_timeout=timeout
        )


@pytest.mark.parametrize(
    "editable_files",
    ["calculator.py", ["../outside.txt"], [".git/config"], [""], [123]],
)
def test_invalid_edit_configuration(workspace, runner, editable_files):
    with pytest.raises(ValueError):
        ToolDispatcher(
            workspace,
            runner,
            test_command=["project-test"],
            editable_files=editable_files,
        )
