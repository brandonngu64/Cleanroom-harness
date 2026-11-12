# Cleanroom architecture

Cleanroom coordinates one coding agent working on a disposable repository copy.
The core harness does not depend on Docker, a target language, or a test framework.

## Module boundaries

| Module | Responsibility | State |
| --- | --- | --- |
| `workspace.py` | Controlled repository file operations | Implemented |
| `runner.py` | Command execution interface, results, and local implementation | Implemented |
| `tools.py` | Validate and dispatch model-requested operations | Implemented |
| `agent.py` | Single-agent loop, normalized model interface, limits, and usage | Implemented |
| `claude.py` | Translate the model interface to the Claude API | Implemented; SDK tested with simulated HTTP |
| `run.py` | Prepare copies, establish a baseline, run the agent, verify, save, clean up | Implemented |
| `report.py` | Save supplied observations, transcript, metrics, and patch | Implemented |
| `cli.py` | Parse user configuration, invoke the coordinator, and report outcomes | Implemented |

## Execution boundary

The harness depends on `Runner.run(command, cwd=..., timeout=...) -> RunResult`.
`Runner` is a Python protocol: implementations provide that method without
needing a shared base class. No plugin registry or execution framework is needed.

The caller supplies an argument sequence, working directory, and timeout.
The result records the command, directory, exit code when available, stdout,
stderr, duration, timeout state, startup error, and output truncation.
`succeeded` means the command exited with code zero without a timeout or startup
error. It does not establish that the coding task was solved or tests were valid.

The runner has no built-in pytest command. Commands such as `python -m pytest`,
`go test ./...`, or an explicitly configured project test script belong in run
configuration. Installed tools and dependencies must be available in the selected
execution environment. Pytest is used to test Cleanroom itself.

## Initial implementation: LocalRunner

`LocalRunner` uses only the Python standard library and executes trusted commands
on the host. It passes arguments with `shell=False`, closes standard input, and
does not install dependencies or change the host's working directory.
Windows batch scripts can still be interpreted by the operating system's shell.

The subprocess timeout stops the launched process. It does not supervise or stop
all descendant processes. Stdout and stderr are redirected to temporary files,
avoiding pipe waits caused by descendants retaining handles. Each returned stream
is limited to 64 KiB by default, with UTF-8 replacement decoding and an explicit
truncation flag. Temporary disk usage is not capped by this return limit.

This implementation inherits the host user's permissions and environment. It does
not isolate filesystem/network access or impose CPU and memory limits. A future
isolated implementation can satisfy the same interface without changing the
agent or dispatcher. Docker is optional, not an architectural requirement.

## Tool boundary

`get_tool_definitions()` returns model-independent descriptions and JSON parameter
schemas for `list_files`, `read_file`, `replace_text`, and `run_tests`. The future
model adapter translates those definitions into its API's request format.

`ToolDispatcher` validates tool names, required string arguments, and unexpected
arguments before dispatching. The schemas and validation share the same argument
definitions. File errors and invalid requests become `ToolResult.error`; unexpected
programming errors propagate so the run coordinator can record a harness failure.
Result data is JSON-serializable, including command results.

The caller supplies an explicit editable-file list; the default permits no edits.
Approved file paths are resolved at setup, and each edit resolves its target again
before checking permission and calling Workspace. Reads still follow Workspace's
existing access policy. This is a file-edit policy, not complete isolation from
symlink races, hard links, or locally executed repository code.

The test command, runner, timeout, and working directory are configured by the
harness. `run_tests` accepts no model arguments. Nonzero command exit codes are
normal tool results with `succeeded = false`; startup failures and timeouts are
tool errors that preserve the command result. Tests cover an alternate recording
runner and the actual local implementation.

Workspace listings and reads currently return their complete results. Before
supporting large repositories, add bounded or paginated file tools alongside the
agent's conversation and token limits.

## Single-agent loop

`CodingAgent` accepts a `ModelClient` and a configured `ToolDispatcher`.
`ModelClient.complete()` receives the conversation, tool definitions, an output
token allowance, and a request timeout. It returns a normalized `ModelReply`.
There is no model SDK dependency in the loop. The Claude API adapter is a separate
component, tested using the actual SDK with simulated HTTP. No live model call
has been verified yet.

Messages represent system/user text, assistant responses with tool calls, and
tool results matched by call ID. An adapter can preserve original provider
content in JSON-compatible `provider_data`, which the loop stores unchanged.
The adapter translates these messages into its provider's required format.
Model requests receive copies of the conversation so they cannot mutate its log.

Each response is validated before any requested tool is dispatched. Multiple
calls are processed sequentially. Duplicate call IDs and malformed responses stop
the loop. Tool errors are returned to the model so it can correct its next request.
Truncated or refused responses never cause tool execution. Operations skipped by
a limit receive explicit error results, preserving the transcript and call IDs.

`AgentLimits` bounds attempted model requests, dispatched tool calls, requested
output tokens per response, observed total token usage, and elapsed time. Failed
tool requests count toward the tool limit. After using its last tool allowance,
the model may still provide a final answer; further tools are skipped.

Time checks occur around model requests and between tools. The model adapter must
honor the remaining request timeout. A tool already executing uses its own limit
and can exceed the agent's elapsed-time allowance. Total input/output usage is
known after a response, so the usage limit may be exceeded by one request. These
are cooperative controls, not hard cancellation or spending guarantees. Adapters
report total processed input, including cached input where applicable; API costs
are not calculated yet.

`AgentResult` preserves the transcript, last response text, stop reason, request
and tool counts, usage from received responses, duration, and expected model errors.
Unexpected client/dispatcher programming failures still propagate to the future
run coordinator. A `completed` stop reason means the model ended its turn; the
coordinator must perform independent final verification before claiming success.

Tests use scripted model replies and actual workspace/runner operations, including
a calculator fix that changes a failing prewritten check to a passing one. This
verifies control flow without a model API call or an autonomous reasoning claim.

## Claude API adapter

`ClaudeClient` implements the model interface and loads the optional Anthropic
SDK when constructed. The caller selects the model; credentials may come from
`ANTHROPIC_API_KEY` or an explicitly supplied SDK client. Owned clients support
context-manager cleanup; injected clients remain owned by their caller.

System text is passed in the API's system field. Generic tool schemas become
`input_schema` definitions. Assistant tool calls are followed immediately by one
user message containing the complete matching batch of tool results. Original
assistant content is retained in `provider_data`, preserving signed thinking
blocks when the conversation is sent again.

Automatic SDK retries are disabled per request. The remaining request timeout
is forwarded to the transport. API and transport failures become `ModelError`.
Normal completion, tool use, output/context exhaustion, and refusals are mapped
to the loop's stop reasons. Server-tool pause turns and custom stop-sequence
responses are explicitly unsupported by this initial adapter.

Input usage includes ordinary input, cache creation, and cache reads. Native
response metadata is preserved in the transcript. Tests use SDK 2.8.0 and a
simulated HTTP transport, including a complete scripted edit/verify/report flow.
They do not make paid API calls or establish live account/model access.

## Report persistence

`RunReport` holds task/repository/model metadata, optional agent/baseline/final
results, a source revision, execution configuration, a supplied patch, and errors.
`save_report()` creates a new directory with `report.json`, `transcript.json`,
and `changes.patch`. The summary separates model completion from verification
command status. Missing final verification is reported as `not_run`.

Report serialization is checked before creating the directory. Existing output
directories are never overwritten. Disk errors propagate and may leave partial
files; the summary is written last. The writer does not prepare repositories,
run verification, compute patches, or calculate API costs. Those observations
will come from the coordinator and execution configuration.

## Package setup

`pyproject.toml` installs the `harness` package from `src`, configures pytest's
source path, and declares optional `dev` and `claude` extras. The core uses the
standard library. The `cleanroom` entry point and `python -m harness` both call
`harness.cli.main`.
The README documents installation, programmatic use, current limits, and the
remaining integration work.

## Run coordination

`run_task(RunConfig(...), client, output_directory=..., runner=...)` owns a complete
execution. It accepts a configured model client and runner without importing a
provider SDK. The default execution implementation is `LocalRunner`.

The initial source contract is a clean local Git working-tree root with a recorded
commit. Tracked/staged changes and untracked files are rejected; ignored local
files are omitted. Existing tracked regular files are supported. Symlinks and
submodules are rejected, and Git LFS hydration is not implemented. Editable-file
configuration is explicit and cannot approve files outside the repository.

A local clone uses `--no-hardlinks` and `--dissociate` so the disposable snapshot
does not borrow source object files. Git helpers discard inherited Git path/config
injection overrides, use an empty hooks directory, restrict protocols to local
files, and disable external diff/text conversion helpers. Read-only source status
checks respect the user's configuration. Effective `core.autocrlf` and `core.eol`
are captured and stored in the disposable clone and report; remaining disposable
Git operations ignore global/system configuration. This keeps clean-tree checks
and saved patches compatible with local or global line-ending settings. The
source commit is checked out detached. This is preparation for trusted repositories,
not an execution security boundary. Git and subprocess working directories must
fit the platform's working-directory limits; the OS temp location is used by default.

Separate physical copies are prepared for baseline execution, the coding agent,
and final verification. Baseline command side effects do not enter the agent copy.
Final verification starts from the recorded commit and receives only approved
regular files from the agent's copy. Unapproved tracked edits are recorded as a
policy error and excluded from verification and the saved patch. Generated or
untracked files are not transferred. The original repository is never the runner's
working directory and the coordinator does not apply changes back to it.

Patch extraction uses the pristine Git index, regardless of edits to Git metadata
in the agent copy. Patch bytes preserve newline content; external diff drivers and
text conversion are disabled. Tests check the saved patch with `git apply --check`
against the unchanged original commit.

Failed baseline assertions are normal input to the agent. A baseline startup error,
timeout, or missing exit status stops the run before any model request. Agent limits
and model errors still lead to a final verification attempt when preparation and
approved-file transfer remain possible. Unexpected failures record available
observations, errors, and a traceback. Interruptions preserve the available patch
without starting additional verification. An unexpected exception may prevent an
AgentResult from being returned, so its partial token usage/transcript is unavailable.

`RunOutcome` contains the report, artifact directory, any cleanup failure, and a
structured interruption flag for the CLI's exit status.
Reports are persisted before cleanup. A save failure raises `ReportSaveError`
carrying the report and retained scratch directory. Cleanup verifies the owned
directory remains inside its recorded temporary parent before recursive removal.
Cleanup failures preserve the directory and save a warning when possible.

Agent limits apply to the model/tool phase. Baseline/final commands and Git
operations have their own timeouts. Recorded execution duration excludes artifact
writing and cleanup. This pipeline keeps command verification separate from model
completion and does not infer that passing tests proves a task was solved.

## CLI and example

`cleanroom run` takes a local Git root, task, model, new external output directory,
approved editable paths, and execution limits. The test command is passed after
`--` as an argument sequence without shell-string splitting. The CLI chooses the
initial Claude adapter; the agent loop and runner contracts remain independent.
The model ID can come from `CLEANROOM_MODEL`, and the API key comes from
`ANTHROPIC_API_KEY`. Help works without credentials or the optional SDK.
Package installation and the module entry point are verified. The generated
Windows `.exe` launcher is blocked by this machine's group policy; the documented
module invocation remains available without changing the policy.

Exit `0` requires model completion, passing final verification, and no run or
cleanup error. Failed verification or incomplete model work exits `1`; invalid
arguments exit `2`; setup/execution/persistence errors exit `3`; interruption exits
`130`. Output paths and available status/summary information are printed. A report
save failure displays the retained scratch directory. Interruptions during saving
are wrapped with the retained report/workspace rather than losing recovery data.

`examples/buggy_calculator` contains an intentionally broken function, four
prewritten unittest cases, and instructions for making a separate committed Git
fixture. CLI integration tests use this example with scripted model replies and
real Git repositories and subprocess checks. No live model call is implied.

## Run lifecycle

1. Validate the task, source repository, trusted execution configuration, and limits.
2. Prepare a disposable repository copy and record its starting revision.
3. Run baseline tests using the configured runner and command.
4. Start the single coding agent and dispatch its approved operations.
5. Run final verification independently of the agent's completion message.
6. Save the patch, baseline/final results, stop reason, usage, timing, and logs.
7. Clean up temporary resources after preserving artifacts.

The coordinator chooses the runner. The dispatcher exposes a configured test
operation; the model does not get to choose arbitrary host commands, execution
implementations, or execution limits. Trusted evaluation tests and configuration
must be protected from agent edits.

## Development order

Workspace and tests -> runner and tests -> tool dispatcher -> agent loop ->
Claude API adapter -> reporting and run coordination -> CLI -> one end-to-end
calculator demonstration with a live model.
Multiple coding agents, generated tests, and alternate model providers remain
outside the first milestone.
