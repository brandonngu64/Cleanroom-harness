# Cleanroom

Cleanroom is a Python harness for one coding agent working on a software task.
It controls file operations, runs a configured verification command, and records
the conversation and results. The core does not require Docker, a particular
target language, or a particular test framework.

## Current state

Implemented: workspace file access, configurable local command execution, tool
validation and editable-file permissions, the single-agent loop, a Claude API
adapter, and report persistence.

The tests exercise the real Anthropic SDK through simulated HTTP responses and
run real local verification commands. A live model run has not been verified.
The disposable-copy coordinator and command-line interface are still planned.

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

See [the architecture notes](docs/architecture.md) for the module contracts and
remaining work.

## Using the Claude adapter

Install the `claude` extra and set `ANTHROPIC_API_KEY` in your environment.
Select a model ID available to your account. Tests use dummy credentials and
simulated HTTP; they do not need an API key or make paid requests.

Programmatic use requires an existing disposable copy of a trusted repository:

```python
import os
import sys

from harness.agent import CodingAgent
from harness.claude import ClaudeClient
from harness.runner import LocalRunner
from harness.tools import ToolDispatcher
from harness.workspace import Workspace

workspace = Workspace("path/to/a-disposable-copy")
dispatcher = ToolDispatcher(
    workspace,
    LocalRunner(),
    test_command=[sys.executable, "-m", "pytest", "-q"],
    test_timeout=30,
    editable_files=["calculator.py"],
)

with ClaudeClient(model=os.environ["CLEANROOM_MODEL"]) as client:
    result = CodingAgent(client, dispatcher).run("Fix the add function")

print(result.stop_reason)
print(result.final_text)
```

`CLEANROOM_MODEL` is an example environment variable read by this snippet.
The caller supplies the actual model ID to `ClaudeClient`. The test command is
also caller-controlled; pytest above is only an example. Required project tools
and dependencies must already be installed.

The model's completion message is not an independent verification result. The
future run coordinator will perform baseline and final checks automatically.

## Saving a report

```python
from harness.report import RunReport, save_report

report = RunReport(
    task="Fix the add function",
    repository=workspace.root,
    model=os.environ["CLEANROOM_MODEL"],
    agent=result,
)
directory = save_report(report, "runs/example-001")
```

The output directory must be new. It contains:

- `report.json`: run metadata, command outcomes, and agent metrics.
- `transcript.json`: model messages, tool calls/results, and provider content.
- `changes.patch`: the patch supplied by the caller, empty if none was supplied.

Pass `baseline`, `verification`, `source_revision`, `patch`, and execution
`configuration` when those observations are available. Missing verification is
reported as `not_run`. The writer does not create a repository copy, run tests,
or generate a Git diff. Serialization is checked before the output directory is
created. File I/O errors propagate and may leave partial artifacts.

## Current limits and next steps

Local execution uses the host user's permissions and environment. It does not
isolate filesystem/network access, cap CPU or memory, or supervise an entire
process tree. Its output return limit does not cap temporary disk usage.
Agent time and token limits are cooperative; they are not hard cancellation or
spending guarantees. File listings and reads are not yet paginated.

Next: build `run.py` for disposable copies, baseline/final verification, Git
patch capture, and cleanup; then add `cli.py` and a permanent live-model example.
CI and benchmark runs follow the first complete demonstration. The initial
implementation uses one coding agent.
