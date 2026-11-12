# Cleanroom

Cleanroom is a Python harness for one coding agent working on a software task.
It controls file operations, runs a configured verification command, and records
the conversation and results. The core does not require Docker, a particular
target language, or a particular test framework.

## Current state

Implemented: workspace file access, configurable local command execution, tool
validation and editable-file permissions, the single-agent loop, a Claude API
adapter, report persistence, and a coordinator for complete runs on disposable
copies of clean local Git repositories, and a command-line interface.

The tests exercise the real Anthropic SDK through simulated HTTP responses and
run real local verification commands. A live model run has not been verified.
The CLI and bundled calculator example are tested with scripted model replies.

## Development setup

Requires Python 3.10 or later. From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,claude]"
.\.venv\Scripts\python.exe -m pytest
```

On other platforms, use the virtual environment's `bin/python` executable.
The editable installation imports the package as `harness`. Pytest's source path
is configured in `pyproject.toml`, so no command-line `pythonpath` override is
needed. Install `.[dev]` instead to work without Claude support; SDK-specific
tests are then skipped. The core has no third-party runtime dependencies.

The Claude adapter is tested with Anthropic SDK 2.8.0. Its dependency is optional
and constrained to compatible 2.x releases. No model ID is hard-coded.

## Components

| File | Responsibility |
| --- | --- |
| `src/harness/workspace.py` | Read, list, and edit repository files |
| `src/harness/runner.py` | Execute configured commands and capture outcomes |
| `src/harness/tools.py` | Validate requests and enforce editable-file permissions |
| `src/harness/agent.py` | Drive the conversation, tool calls, limits, and usage |
| `src/harness/claude.py` | Translate the model interface to the Claude Messages API |
| `src/harness/report.py` | Save the summary, transcript, and supplied patch |
| `src/harness/run.py` | Prepare copies, baseline, invoke the agent, verify, save, and clean up |
| `src/harness/cli.py` | Parse terminal arguments, invoke the coordinator, and report outcomes |

See [the architecture notes](docs/architecture.md) for the module contracts and
remaining work.

## Command-line use

Install or reinstall the editable package to create the `cleanroom` entry point:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,claude]"
.\.venv\Scripts\python.exe -m harness --help
```

`cleanroom run` and `python -m harness run` invoke the same implementation.
Managed Windows policies can block generated console launchers; the module form
above is verified in this development environment and does not use that launcher.
Set `ANTHROPIC_API_KEY` in the environment. Supply a model with `--model`, or set
`CLEANROOM_MODEL`. Help does not require credentials or the optional SDK.

From the Cleanroom repository root, with a separate clean target repository:

```powershell
$python = (Resolve-Path .\.venv\Scripts\python.exe).Path
& $python -m harness run --repo ..\target-repo --task "Fix the add function" --model "<model-id>" --output .\runs\run-001 --editable calculator.py -- $python -m pytest -q
```

All Cleanroom options go before `--`. The test executable and its arguments go
after it and are passed as separate arguments. The command and installed tools
are your choice; pytest is an example. Repeat `--editable PATH` for each approved
file. Without `--editable`, the agent has read-only file tools.

`--output` is required and must name a new directory outside the target repository.
Use `--test-timeout` and `--git-timeout` for command/preparation limits. Agent
controls are `--max-iterations`, `--max-tool-calls`, `--max-output-tokens`,
`--max-total-tokens`, and `--max-seconds`. Their cooperative limits are described
below and in the architecture notes.

The terminal prints baseline status, the agent's stop reason, final verification
status, artifact paths, and the model's final text when available. Exit codes:

| Code | Meaning |
| --- | --- |
| `0` | Model completed, final verification passed, and no run/cleanup error was recorded |
| `1` | Verification failed or the agent stopped before completion |
| `2` | Invalid command-line arguments or configuration |
| `3` | Setup, preparation, API, verification execution, artifact, or cleanup error |
| `130` | Interrupted |

These describe execution and verification, not proof that the coding task was
solved. API keys are read from the environment and are not command-line arguments
or report configuration fields. This initial CLI uses the optional Claude adapter;
the programmatic interfaces remain provider-independent.

## Calculator example

[The bundled calculator](examples/buggy_calculator/README.md) deliberately contains
a subtraction bug and four existing unittest cases. Its README shows how to copy
it into a separate Git repository, commit the fixture, and run the CLI against it.
The test suite exercises the same example with a scripted model and real Git/test
commands. A live model run remains to be verified with configured credentials.

## Programmatic use

Install the `claude` extra and set `ANTHROPIC_API_KEY` in your environment.
Select a model ID available to your account. Tests use dummy credentials and
simulated HTTP; they do not need an API key or make paid requests.

Programmatic use requires Git and a clean, committed local repository. The
coordinator prepares disposable copies automatically:

```python
import os
import sys

from harness.claude import ClaudeClient
from harness.report import command_status
from harness.run import RunConfig, run_task

config = RunConfig(
    repository="path/to/a-clean-git-repository",
    task="Fix the add function",
    model=os.environ["CLEANROOM_MODEL"],
    test_command=[sys.executable, "-m", "pytest", "-q"],
    test_timeout=30,
    editable_files=["calculator.py"],
)

with ClaudeClient(model=config.model) as client:
    outcome = run_task(config, client, output_directory="runs/example-001")

print(command_status(outcome.report.verification))
print(outcome.report.error)
print(outcome.output_directory)
```

`CLEANROOM_MODEL` is an example environment variable read by this snippet.
The caller supplies the actual model ID to `ClaudeClient`. The test command is
also caller-controlled; pytest above is only an example. Required project tools
and dependencies must already be installed.

`run_task` runs the baseline, invokes the agent, then verifies a fresh copy of
the recorded commit with only approved source files transferred. The final
verification does not reuse tests or configuration changed in the agent's copy.
Unapproved tracked changes are excluded and recorded as a policy error. Source
files in the original repository are not updated; review the saved patch.

The source must have a clean index and working tree, including untracked files.
Ignored local files are not copied. Editable files must already be tracked regular
files. This version rejects symlinks and submodules and does not hydrate Git LFS
objects. The output directory must be new and outside the source repository.

The default runner is `LocalRunner`; pass another implementation using
`runner=...`. This copy/verification policy is not isolation from locally executed
code, which still runs with the selected runner's permissions.

## Saving a report

`run_task` saves its report automatically. The new output directory contains:

- `report.json`: run metadata, command outcomes, and agent metrics.
- `transcript.json`: model messages, tool calls/results, and provider content.
- `changes.patch`: a Git patch containing approved changes, empty if none exist.

The coordinator supplies baseline/final command results, the source commit,
approved Git diff, and execution configuration. Model completion and verification
command status remain separate observations. Missing verification is `not_run`.
`RunReport` and `save_report` are also available for callers supplying their own
observations outside the coordinator.

Artifacts are saved before temporary copies are cleaned up. If saving fails,
`ReportSaveError` exposes the in-memory report and retained workspace for recovery.
Cleanup failures are returned in `RunOutcome.cleanup_error` and recorded in
`cleanup_error.txt` when possible. A caught interruption saves available changes
and failure details, skips further verification, and attempts cleanup.
`RunOutcome.interrupted` and report configuration record interruptions caught
during execution. If report saving itself is interrupted, `ReportSaveError`
retains the report/workspace and the CLI returns `130`.

## Current limits and next steps

Local execution uses the host user's permissions and environment. It does not
isolate filesystem/network access, cap CPU or memory, or supervise an entire
process tree. Its output return limit does not cap temporary disk usage.
Agent time and token limits are cooperative; they are not hard cancellation or
spending guarantees. File listings and reads are not yet paginated.

Next: verify the bundled example with a live model, then add CI and benchmark
runs. The initial
implementation uses one coding agent.
