"""Command-line access to the single-agent run coordinator."""

import argparse
from collections.abc import Sequence
import os
from pathlib import Path
import sys

from .agent import AgentLimits
from .claude import ClaudeClient
from .report import command_status
from .run import ReportSaveError, RunConfig, RunOutcome, run_task


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cleanroom",
        description="Run a coding agent on disposable copies of a trusted local Git repository.",
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run",
        help="Edit an approved set of files and independently verify the result.",
        allow_abbrev=False,
        epilog="Put all Cleanroom options before --, followed by the test executable and its arguments.",
    )
    run.add_argument(
        "--repo", required=True, type=Path, help="Clean, committed local Git root."
    )
    run.add_argument("--task", required=True, help="Coding task for the agent.")
    run.add_argument(
        "--model",
        default=os.environ.get("CLEANROOM_MODEL"),
        help="Claude model ID; defaults to CLEANROOM_MODEL.",
    )
    run.add_argument(
        "--output",
        required=True,
        type=Path,
        help="New artifact directory outside the source repository.",
    )
    run.add_argument(
        "--editable",
        action="append",
        default=[],
        metavar="PATH",
        help="Approved relative file path; repeat for multiple files. Default: read-only.",
    )
    run.add_argument("--test-timeout", type=float, default=60.0, metavar="SECONDS")
    run.add_argument("--git-timeout", type=float, default=30.0, metavar="SECONDS")
    defaults = AgentLimits()
    run.add_argument("--max-iterations", type=int, default=defaults.max_iterations)
    run.add_argument("--max-tool-calls", type=int, default=defaults.max_tool_calls)
    run.add_argument(
        "--max-output-tokens", type=int, default=defaults.max_output_tokens
    )
    run.add_argument("--max-total-tokens", type=int, default=defaults.max_total_tokens)
    run.add_argument("--max-seconds", type=float, default=defaults.max_seconds)
    run.add_argument(
        "test_command",
        nargs=argparse.REMAINDER,
        metavar="TEST_COMMAND",
        help="Test executable and arguments, after --; no implicit shell.",
    )
    return parser


def _exit_code(outcome: RunOutcome) -> int:
    if outcome.interrupted:
        return 130
    report = outcome.report
    if report.error is not None or outcome.cleanup_error is not None:
        return 3
    if report.agent is None or command_status(report.verification) in (
        "not_run",
        "unverified",
        "error",
        "timed_out",
    ):
        return 3
    if report.agent.stop_reason != "completed" or not report.verification.succeeded:
        return 1
    return 0


def _print_outcome(outcome: RunOutcome) -> None:
    report = outcome.report
    print(f"Baseline: {command_status(report.baseline)}")
    print(f"Agent: {report.agent.stop_reason if report.agent else 'not_started'}")
    print(f"Verification: {command_status(report.verification)}")
    print(f"Report: {outcome.output_directory / 'report.json'}")
    print(f"Patch: {outcome.output_directory / 'changes.patch'}")
    if report.agent is not None and report.agent.final_text:
        print()
        print(report.agent.final_text)
    if report.error is not None:
        print(f"Run error: {report.error}", file=sys.stderr)
    if outcome.cleanup_error is not None:
        print(f"Cleanup error: {outcome.cleanup_error}", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    options = parser.parse_args(argv)
    command = options.test_command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("Provide a test executable and arguments after --")
    if options.model is None:
        parser.error("Provide --model or set CLEANROOM_MODEL")
    try:
        limits = AgentLimits(
            max_iterations=options.max_iterations,
            max_tool_calls=options.max_tool_calls,
            max_output_tokens=options.max_output_tokens,
            max_total_tokens=options.max_total_tokens,
            max_seconds=options.max_seconds,
        )
        config = RunConfig(
            repository=options.repo,
            task=options.task,
            model=options.model,
            test_command=command,
            editable_files=options.editable,
            test_timeout=options.test_timeout,
            git_timeout=options.git_timeout,
            agent_limits=limits,
        )
    except ValueError as exc:
        parser.error(str(exc))

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key.strip():
        print(
            "cleanroom: Set ANTHROPIC_API_KEY before running the Claude adapter.",
            file=sys.stderr,
        )
        return 3

    try:
        with ClaudeClient(model=config.model, api_key=api_key) as client:
            outcome = run_task(config, client, output_directory=options.output)
    except ReportSaveError as exc:
        print(
            f"cleanroom: Report saving failed: {exc.__cause__ or exc}", file=sys.stderr
        )
        if exc.workspace is not None:
            print(f"Retained workspace: {exc.workspace}", file=sys.stderr)
        return 130 if isinstance(exc.__cause__, KeyboardInterrupt) else 3
    except KeyboardInterrupt:
        print(
            f"cleanroom: Interrupted. Check {options.output} for saved or partial artifacts.",
            file=sys.stderr,
        )
        return 130
    except Exception as exc:
        print(f"cleanroom: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3

    _print_outcome(outcome)
    return _exit_code(outcome)
