"""Model-independent tool definitions and dispatch for a single workspace."""

from collections.abc import Sequence
from dataclasses import asdict, dataclass
import math

from .runner import Runner
from .workspace import Workspace

# Each argument description also defines a required string argument.
_TOOL_SPECS = {
    "list_files": ("List files in the workspace, excluding Git metadata.", {}),
    "read_file": (
        "Read a UTF-8 text file inside the workspace.",
        {"path": "Path to the file, relative to the workspace."},
    ),
    "replace_text": (
        "Replace exactly one occurrence of text in an approved editable file.",
        {
            "path": "Path to the file, relative to the workspace.",
            "old": "Nonempty text that must occur exactly once.",
            "new": "Replacement text; may be empty to remove the old text.",
        },
    ),
    "run_tests": (
        "Run the harness-configured test command with its configured timeout.",
        {},
    ),
}


def get_tool_definitions() -> list[dict[str, object]]:
    """Return fresh definitions; a provider adapter chooses its API format."""
    return [
        {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    argument: {"type": "string", "description": detail}
                    for argument, detail in arguments.items()
                },
                "required": list(arguments),
                "additionalProperties": False,
            },
        }
        for name, (description, arguments) in _TOOL_SPECS.items()
    ]


@dataclass(frozen=True)
class ToolResult:
    name: str
    data: dict[str, object] | None = None
    error: str | None = None

    @property
    def is_error(self) -> bool:
        return self.error is not None


class ToolDispatcher:
    """Validate requests and execute only the four approved operations.

    File edits are read-only by default: callers must approve editable files.
    This permission check supplements Workspace path checks; it is not a sandbox.
    Test commands and timeouts come from trusted harness configuration.
    """

    def __init__(
        self,
        workspace: Workspace,
        runner: Runner,
        *,
        test_command: Sequence[str],
        test_timeout: float = 60.0,
        editable_files: Sequence[str] = (),
    ):
        if isinstance(test_command, (str, bytes)):
            raise ValueError("test_command must be a sequence of arguments")
        command = tuple(test_command)
        if (
            not command
            or any(not isinstance(argument, str) for argument in command)
            or not command[0]
            or any("\0" in argument for argument in command)
        ):
            raise ValueError("test_command must contain valid string arguments")

        if (
            not isinstance(test_timeout, (int, float))
            or isinstance(test_timeout, bool)
            or not math.isfinite(test_timeout)
            or test_timeout <= 0
        ):
            raise ValueError("test_timeout must be a positive, finite number")

        if isinstance(editable_files, (str, bytes)):
            raise ValueError("editable_files must be a sequence of file paths")

        self._workspace = workspace
        self._runner = runner
        self._test_command = command
        self._test_timeout = test_timeout
        approved = set()
        for path in editable_files:
            if not isinstance(path, str) or not path or "\0" in path:
                raise ValueError("Editable file paths must be nonempty strings")
            target = (workspace.root / path).resolve()
            if not target.is_relative_to(workspace.root):
                raise ValueError("Editable files must be inside the workspace")
            relative = target.relative_to(workspace.root)
            if any(part.casefold() == ".git" for part in relative.parts):
                raise ValueError("Git metadata cannot be editable")
            approved.add(target)
        self._editable_files = frozenset(approved)

    def dispatch(self, name: str, arguments: object) -> ToolResult:
        if not isinstance(name, str):
            return ToolResult(name="", error="Tool name must be a string")
        if name not in _TOOL_SPECS:
            return ToolResult(name=name, error=f"Unknown tool: {name}")

        try:
            args = self._validate_arguments(name, arguments)

            if name == "list_files":
                return ToolResult(name, data={"files": self._workspace.list_files()})

            if name == "read_file":
                content = self._workspace.read_file(args["path"])
                return ToolResult(name, data={"path": args["path"], "content": content})

            if name == "replace_text":
                target = (self._workspace.root / args["path"]).resolve()
                if target not in self._editable_files:
                    raise PermissionError("Editing this file is not allowed")
                self._workspace.replace_text(args["path"], args["old"], args["new"])
                return ToolResult(name, data={"path": args["path"], "updated": True})

            if name == "run_tests":
                return self._run_tests()

            raise RuntimeError(f"No handler implemented for tool: {name}")

        except (OSError, UnicodeError, ValueError) as exc:
            # Expected request/operation failures can be returned to the model.
            # Unexpected programming errors propagate to the run coordinator.
            return ToolResult(name=name, error=str(exc))

    def _run_tests(self) -> ToolResult:
        # Model arguments cannot change the command, directory, or timeout.
        result = self._runner.run(
            self._test_command,
            cwd=self._workspace.root,
            timeout=self._test_timeout,
        )
        data = asdict(result)
        data["command"] = list(result.command)
        data["cwd"] = str(result.cwd)
        data["succeeded"] = result.succeeded
        error = result.error
        if result.timed_out and error is None:
            error = "Test command timed out"
        return ToolResult("run_tests", data=data, error=error)

    def _validate_arguments(self, name: str, arguments: object) -> dict[str, str]:
        if not isinstance(arguments, dict) or any(
            not isinstance(key, str) for key in arguments
        ):
            raise ValueError("Arguments must be a JSON object with string keys")

        expected = set(_TOOL_SPECS[name][1])
        missing = expected - arguments.keys()
        unexpected = arguments.keys() - expected
        if missing:
            raise ValueError(
                f"Missing required arguments: {', '.join(sorted(missing))}"
            )
        if unexpected:
            raise ValueError(f"Unexpected arguments: {', '.join(sorted(unexpected))}")
        for argument in expected:
            if not isinstance(arguments[argument], str):
                raise ValueError(f"Argument '{argument}' must be a string")
        return arguments.copy()
