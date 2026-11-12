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
| `run.py` | Prepare a copy, establish a baseline, run the agent, verify, save, clean up | Planned |
| `report.py` | Save supplied observations, transcript, metrics, and patch | Implemented |
| `cli.py` | Parse user configuration and invoke the coordinator | Planned |

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
standard library. There is no console entry point until the CLI exists.
The README documents installation, programmatic use, current limits, and the
remaining integration work.

## Planned run lifecycle

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
