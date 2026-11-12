from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path

import pytest

from harness.agent import AgentLimits, AgentResult, Message
from harness.report import RunReport, command_status, save_report
from harness.runner import RunResult


def command_result(tmp_path, **changes):
    result = RunResult(
        command=("project-test",),
        cwd=tmp_path,
        exit_code=0,
        stdout="ok\n",
        stderr="",
        duration_seconds=0.1,
    )
    return replace(result, **changes)


def agent_result(**changes):
    result = AgentResult(
        stop_reason="completed",
        final_text="Done",
        iterations=1,
        tool_calls=0,
        input_tokens=10,
        output_tokens=5,
        duration_seconds=0.2,
        messages=(Message("user", "Fix"), Message("assistant", "Done")),
    )
    return replace(result, **changes)


def test_saves_summary_transcript_and_exact_utf8_patch(tmp_path):
    patch = "--- a/café.py\n+++ b/café.py\n-old\n+new\n"
    report = RunReport(
        task="Fix addition",
        repository=tmp_path / "source",
        model="test-model",
        source_revision="abc123",
        agent=agent_result(),
        baseline=command_result(tmp_path, exit_code=1),
        verification=command_result(tmp_path),
        patch=patch,
        configuration={"limits": AgentLimits(), "runner": "LocalRunner"},
    )
    saved = save_report(report, tmp_path / "artifacts" / "run-1")
    summary = json.loads((saved / "report.json").read_text(encoding="utf-8"))
    transcript = json.loads((saved / "transcript.json").read_text(encoding="utf-8"))

    assert set(path.name for path in saved.iterdir()) == {
        "report.json",
        "transcript.json",
        "changes.patch",
    }
    assert summary["schema_version"] == 1
    assert summary["repository"] == str(tmp_path / "source")
    assert summary["source_revision"] == "abc123"
    assert summary["baseline_status"] == "failed"
    assert summary["verification_status"] == "passed"
    assert summary["agent"]["input_tokens"] == 10
    assert "messages" not in summary["agent"]
    assert "patch" not in summary
    assert transcript[-1]["text"] == "Done"
    assert (saved / "changes.patch").read_bytes() == patch.encode("utf-8")
    assert summary["configuration"]["limits"]["max_iterations"] == 20
    assert (
        datetime.fromisoformat(summary["created_at"]).utcoffset().total_seconds() == 0
    )
    assert report.agent.messages[-1].text == "Done"


@pytest.mark.parametrize(
    "changes, expected",
    [
        ({}, "passed"),
        ({"exit_code": 1}, "failed"),
        ({"exit_code": None}, "unverified"),
        ({"exit_code": None, "timed_out": True}, "timed_out"),
        ({"exit_code": None, "error": "Could not start"}, "error"),
    ],
)
def test_command_outcomes_remain_distinct(tmp_path, changes, expected):
    assert command_status(command_result(tmp_path, **changes)) == expected


def test_absent_command_was_not_run():
    assert command_status(None) == "not_run"


def test_model_completion_does_not_imply_verification(tmp_path):
    saved = save_report(
        RunReport(
            task="Fix", repository=tmp_path, model="test-model", agent=agent_result()
        ),
        tmp_path / "run",
    )
    summary = json.loads((saved / "report.json").read_text(encoding="utf-8"))
    assert summary["agent"]["stop_reason"] == "completed"
    assert summary["verification_status"] == "not_run"


def test_can_record_failure_before_agent_execution(tmp_path):
    saved = save_report(
        RunReport(
            task="Fix",
            repository=tmp_path,
            model="test-model",
            error="Preparation failed",
        ),
        tmp_path / "run",
    )
    summary = json.loads((saved / "report.json").read_text(encoding="utf-8"))
    assert summary["agent"] is None
    assert summary["error"] == "Preparation failed"
    assert summary["verification_status"] == "not_run"
    assert json.loads((saved / "transcript.json").read_text(encoding="utf-8")) == []


def test_existing_directory_is_never_overwritten(tmp_path):
    target = tmp_path / "run"
    target.mkdir()
    marker = target / "keep.txt"
    marker.write_text("existing contents", encoding="utf-8")
    with pytest.raises(FileExistsError):
        save_report(
            RunReport(task="Fix", repository=tmp_path, model="test-model"), target
        )
    assert marker.read_text(encoding="utf-8") == "existing contents"
    assert set(target.iterdir()) == {marker}


@pytest.mark.parametrize(
    "configuration", [{"unsupported": object()}, {"number": float("nan")}]
)
def test_serialization_failure_creates_no_output(tmp_path, configuration):
    target = tmp_path / "run"
    report = RunReport(
        task="Fix", repository=tmp_path, model="test-model", configuration=configuration
    )
    with pytest.raises((TypeError, ValueError)):
        save_report(report, target)
    assert not target.exists()


def test_disk_write_errors_propagate_without_claiming_a_saved_summary(
    tmp_path, monkeypatch
):
    target = tmp_path / "run"
    original_open = Path.open

    def failing_open(path, *args, **kwargs):
        if path.name == "changes.patch":
            raise OSError("Simulated disk failure")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", failing_open)
    with pytest.raises(OSError, match="disk failure"):
        save_report(
            RunReport(task="Fix", repository=tmp_path, model="test-model"), target
        )
    assert (target / "transcript.json").exists()
    assert not (target / "report.json").exists()


@pytest.mark.parametrize(
    "changes", [{"task": ""}, {"model": ""}, {"repository": ""}, {"patch": 123}]
)
def test_invalid_report_fields(changes):
    fields = {"task": "Fix", "repository": "repo", "model": "test-model", **changes}
    with pytest.raises(ValueError):
        RunReport(**fields)


@pytest.mark.parametrize("duration", [-1, True, float("inf"), float("nan")])
def test_invalid_total_duration(duration):
    with pytest.raises(ValueError):
        RunReport(
            task="Fix", repository="repo", model="test-model", duration_seconds=duration
        )
