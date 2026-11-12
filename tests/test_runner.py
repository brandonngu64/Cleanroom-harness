from pathlib import Path
import sys

import pytest

from harness.runner import LocalRunner, Runner


@pytest.fixture
def runner() -> Runner:
    return LocalRunner()


def python_command(code: str, *arguments: str) -> list[str]:
    return [sys.executable, "-I", "-B", "-c", code, *arguments]


def test_runs_in_requested_directory(runner, tmp_path):
    root = tmp_path / "project with spaces"
    root.mkdir()
    (root / "input.txt").write_text("project contents", encoding="utf-8")
    command = python_command(
        "from pathlib import Path; "
        "print(Path.cwd()); "
        "print(Path('input.txt').read_text(encoding='utf-8'))"
    )

    result = runner.run(command, cwd=root)

    assert result.succeeded
    assert result.exit_code == 0
    assert result.command == tuple(command)
    assert result.cwd == root.resolve()
    lines = result.stdout.splitlines()
    assert Path(lines[0]) == root.resolve()
    assert lines[1] == "project contents"
    assert result.stderr == ""
    assert result.duration_seconds >= 0
    assert not result.timed_out
    assert result.error is None
    assert not result.output_truncated


def test_nonzero_exit_is_a_command_failure(runner, tmp_path):
    result = runner.run(
        python_command(
            "import sys; print('details', file=sys.stderr); sys.exit(3)"
        ),
        cwd=tmp_path,
    )

    assert result.exit_code == 3
    assert result.stderr.strip() == "details"
    assert not result.succeeded
    assert not result.timed_out
    assert result.error is None


def test_missing_program_is_a_start_error(runner, tmp_path):
    result = runner.run([str(tmp_path / "missing-program")], cwd=tmp_path)

    assert result.exit_code is None
    assert result.error
    assert not result.succeeded
    assert not result.timed_out


def test_timeout_preserves_partial_output(runner, tmp_path):
    result = runner.run(
        python_command(
            "import time; print('started', flush=True); time.sleep(10)"
        ),
        cwd=tmp_path,
        timeout=1.0,
    )

    assert result.timed_out
    assert result.exit_code is None
    assert not result.succeeded
    assert result.error is None
    assert result.stdout.strip() == "started"


def test_standard_input_is_closed(runner, tmp_path):
    result = runner.run(
        python_command("import sys; print(repr(sys.stdin.read()))"),
        cwd=tmp_path,
    )

    assert result.succeeded
    assert result.stdout.strip() == "''"


def test_arguments_are_passed_literally(runner, tmp_path):
    argument = "spaces ; & | $ are data"
    result = runner.run(
        python_command("import sys; print(sys.argv[1])", argument),
        cwd=tmp_path,
    )

    assert result.succeeded
    assert result.stdout.strip() == argument


def test_large_output_is_truncated_on_both_streams(tmp_path):
    runner = LocalRunner(max_output_bytes=64)
    result = runner.run(
        python_command(
            "import sys; "
            "sys.stdout.buffer.write(b'o' * 200000); "
            "sys.stderr.buffer.write(b'e' * 200000)"
        ),
        cwd=tmp_path,
    )

    assert result.succeeded
    assert result.stdout == "o" * 64
    assert result.stderr == "e" * 64
    assert result.output_truncated


def test_output_at_limit_is_not_truncated(tmp_path):
    runner = LocalRunner(max_output_bytes=4)
    result = runner.run(
        python_command("import sys; sys.stdout.buffer.write(b'abcd')"),
        cwd=tmp_path,
    )

    assert result.succeeded
    assert result.stdout == "abcd"
    assert not result.output_truncated


def test_invalid_utf8_output_is_replaced(runner, tmp_path):
    result = runner.run(
        python_command("import sys; sys.stdout.buffer.write(b'\\xff')"),
        cwd=tmp_path,
    )

    assert result.succeeded
    assert result.stdout == "\ufffd"


@pytest.mark.parametrize(
    "command",
    [[], "python --version", b"python", [""], [123], ["python", "\0"]],
)
def test_invalid_commands_are_rejected(runner, tmp_path, command):
    with pytest.raises(ValueError):
        runner.run(command, cwd=tmp_path)


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), True])
def test_invalid_timeouts_are_rejected(runner, tmp_path, timeout):
    with pytest.raises(ValueError):
        runner.run(python_command("pass"), cwd=tmp_path, timeout=timeout)


@pytest.mark.parametrize("existing_file", [False, True])
def test_invalid_working_directory(runner, tmp_path, existing_file):
    root = tmp_path / "invalid"
    if existing_file:
        root.write_text("Not a directory", encoding="utf-8")

    with pytest.raises(ValueError):
        runner.run(python_command("pass"), cwd=root)


@pytest.mark.parametrize("limit", [0, -1, 1.5, True])
def test_invalid_output_limits(limit):
    with pytest.raises(ValueError):
        LocalRunner(max_output_bytes=limit)
