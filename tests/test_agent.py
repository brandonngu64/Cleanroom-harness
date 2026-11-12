from copy import deepcopy
from dataclasses import asdict
import json
import sys

import pytest

import harness.agent as agent_module
from harness.agent import (
    AgentLimits,
    CodingAgent,
    ModelError,
    ModelReply,
    ToolCall,
)
from harness.runner import LocalRunner
from harness.tools import ToolDispatcher
from harness.workspace import Workspace


class ScriptedClient:
    """Return chosen model replies, without a network connection or model SDK."""

    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    def complete(self, messages, tools, *, max_output_tokens, timeout):
        request = {
            "messages": deepcopy(messages),
            "tools": deepcopy(tools),
            "max_output_tokens": max_output_tokens,
            "timeout": timeout,
        }
        self.requests.append(request)
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        if callable(reply):
            return reply(messages, tools)
        return reply


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def tool_reply(*calls, input_tokens=1, output_tokens=1):
    return ModelReply(
        stop_reason="tool_calls",
        tool_calls=tuple(calls),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def read_call(call_id="read-1", path="calculator.py"):
    return ToolCall(call_id, "read_file", {"path": path})


def edit_call(call_id="edit-1"):
    return ToolCall(
        call_id,
        "replace_text",
        {"path": "calculator.py", "old": "return a - b", "new": "return a + b"},
    )


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "calculator.py").write_text(
        "def add(a, b):\n    return a - b\n", encoding="utf-8"
    )
    (root / "test_calculator.py").write_text(
        "from calculator import add\n"
        "assert add(2, 3) == 5\n"
        "print('checks passed')\n",
        encoding="utf-8",
    )
    return Workspace(root)


@pytest.fixture
def dispatcher(workspace):
    return ToolDispatcher(
        workspace,
        LocalRunner(),
        test_command=[sys.executable, "-B", "test_calculator.py"],
        editable_files=["calculator.py"],
    )


def test_model_completion_is_not_a_claim_that_the_task_was_solved(
    dispatcher, workspace
):
    client = ScriptedClient(
        [ModelReply(text="Finished", input_tokens=2, output_tokens=3)]
    )
    result = CodingAgent(client, dispatcher).run("Fix the add function")

    assert result.stop_reason == "completed"
    assert result.final_text == "Finished"
    assert result.iterations == 1
    assert result.tool_calls == 0
    assert result.input_tokens == 2
    assert result.output_tokens == 3
    assert result.duration_seconds >= 0
    assert result.error is None
    assert "return a - b" in workspace.read_file("calculator.py")
    assert [message.role for message in result.messages] == [
        "system",
        "user",
        "assistant",
    ]
    assert client.requests[0]["messages"][1].text == "Fix the add function"


def test_scripted_calculator_workflow(dispatcher, workspace):
    baseline = dispatcher.dispatch("run_tests", {})
    assert baseline.data["succeeded"] is False
    client = ScriptedClient(
        [
            tool_reply(read_call(), input_tokens=3, output_tokens=4),
            tool_reply(edit_call(), input_tokens=5, output_tokens=6),
            tool_reply(
                ToolCall("tests-1", "run_tests", {}), input_tokens=7, output_tokens=8
            ),
            ModelReply(
                text="Fixed addition; checks passed.", input_tokens=9, output_tokens=10
            ),
        ]
    )

    result = CodingAgent(client, dispatcher).run("Fix the add function")

    assert result.stop_reason == "completed"
    assert result.iterations == 4
    assert result.tool_calls == 3
    assert result.input_tokens == 24
    assert result.output_tokens == 28
    assert "return a + b" in workspace.read_file("calculator.py")
    tool_messages = [message for message in result.messages if message.role == "tool"]
    assert [message.tool_call_id for message in tool_messages] == [
        "read-1",
        "edit-1",
        "tests-1",
    ]
    outcome = json.loads(tool_messages[-1].text)
    assert outcome["data"]["succeeded"] is True
    assert outcome["data"]["stdout"].strip() == "checks passed"
    json.dumps(asdict(result))


def test_multiple_calls_have_matching_ordered_results(dispatcher):
    client = ScriptedClient(
        [
            tool_reply(read_call(), ToolCall("list-1", "list_files", {})),
            ModelReply(text="Inspected files"),
        ]
    )
    result = CodingAgent(client, dispatcher).run("Inspect the repository")

    assert result.stop_reason == "completed"
    assert result.tool_calls == 2
    messages = client.requests[1]["messages"]
    assert [message.role for message in messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "tool",
    ]
    assert [message.tool_call_id for message in messages[-2:]] == ["read-1", "list-1"]
    assert all(not message.is_error for message in messages[-2:])


@pytest.mark.parametrize(
    "call",
    [
        read_call(path="missing.txt"),
        ToolCall("unknown-1", "shell", {"command": "anything"}),
        ToolCall("invalid-1", "read_file", {"path": 123}),
    ],
)
def test_tool_errors_are_sent_back_to_the_model(dispatcher, call):
    client = ScriptedClient([tool_reply(call), ModelReply(text="Received the error")])
    result = CodingAgent(client, dispatcher).run("Inspect the repository")

    assert result.stop_reason == "completed"
    message = client.requests[1]["messages"][-1]
    assert message.role == "tool"
    assert message.tool_call_id == call.id
    assert message.is_error
    assert json.loads(message.text)["error"]


def test_iteration_limit_preserves_last_tool_results(dispatcher):
    client = ScriptedClient([tool_reply(read_call())])
    result = CodingAgent(client, dispatcher, limits=AgentLimits(max_iterations=1)).run(
        "Inspect"
    )

    assert result.stop_reason == "iteration_limit"
    assert result.iterations == 1
    assert result.tool_calls == 1
    assert result.messages[-1].role == "tool"
    assert len(client.requests) == 1


def test_tool_limit_skips_remaining_calls_in_a_batch(dispatcher, workspace):
    client = ScriptedClient([tool_reply(read_call(), edit_call())])
    result = CodingAgent(client, dispatcher, limits=AgentLimits(max_tool_calls=1)).run(
        "Fix"
    )

    assert result.stop_reason == "tool_limit"
    assert result.tool_calls == 1
    assert "return a - b" in workspace.read_file("calculator.py")
    assert result.messages[-1].tool_call_id == "edit-1"
    assert result.messages[-1].is_error
    assert "Not executed" in result.messages[-1].text


def test_model_can_finish_after_using_its_last_allowed_tool(dispatcher):
    client = ScriptedClient(
        [tool_reply(read_call()), ModelReply(text="Done inspecting")]
    )
    result = CodingAgent(client, dispatcher, limits=AgentLimits(max_tool_calls=1)).run(
        "Inspect"
    )
    assert result.stop_reason == "completed"
    assert result.tool_calls == 1


def test_failed_tool_requests_also_consume_the_tool_budget(dispatcher):
    client = ScriptedClient(
        [tool_reply(read_call(path="missing.txt"), read_call("read-2"))]
    )
    result = CodingAgent(client, dispatcher, limits=AgentLimits(max_tool_calls=1)).run(
        "Inspect"
    )
    assert result.stop_reason == "tool_limit"
    assert result.tool_calls == 1
    assert "Not executed" in result.messages[-1].text


def test_usage_limit_prevents_tool_side_effects(dispatcher, workspace):
    client = ScriptedClient([tool_reply(edit_call(), input_tokens=3, output_tokens=2)])
    result = CodingAgent(
        client, dispatcher, limits=AgentLimits(max_total_tokens=5)
    ).run("Fix")

    assert result.stop_reason == "token_limit"
    assert result.input_tokens + result.output_tokens == 5
    assert result.tool_calls == 0
    assert "return a - b" in workspace.read_file("calculator.py")


def test_output_allowance_is_clamped_to_remaining_token_budget(dispatcher):
    client = ScriptedClient(
        [
            tool_reply(read_call(), input_tokens=2, output_tokens=3),
            ModelReply(text="Done"),
        ]
    )
    limits = AgentLimits(max_output_tokens=50, max_total_tokens=10)
    result = CodingAgent(client, dispatcher, limits=limits).run("Inspect")

    assert result.stop_reason == "completed"
    assert [request["max_output_tokens"] for request in client.requests] == [10, 5]


def test_actual_usage_above_the_limit_is_still_recorded(dispatcher):
    client = ScriptedClient([ModelReply(input_tokens=20, output_tokens=2)])
    result = CodingAgent(
        client, dispatcher, limits=AgentLimits(max_total_tokens=10)
    ).run("Inspect")
    assert result.stop_reason == "token_limit"
    assert result.input_tokens == 20
    assert result.output_tokens == 2


@pytest.mark.parametrize("reason", ["output_limit", "refused"])
def test_terminal_model_stops_do_not_execute_requested_tools(
    dispatcher, workspace, reason
):
    client = ScriptedClient([ModelReply(stop_reason=reason, tool_calls=(edit_call(),))])
    result = CodingAgent(client, dispatcher).run("Fix")
    assert result.stop_reason == reason
    assert result.tool_calls == 0
    assert "return a - b" in workspace.read_file("calculator.py")
    assert result.messages[-1].is_error


def test_request_timeout_uses_the_remaining_time(dispatcher, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(agent_module, "perf_counter", clock)

    def first_reply(messages, tools):
        clock.now = 2
        return tool_reply(read_call())

    client = ScriptedClient([first_reply, ModelReply(text="Done")])
    result = CodingAgent(client, dispatcher, limits=AgentLimits(max_seconds=10)).run(
        "Inspect"
    )
    assert result.stop_reason == "completed"
    assert [request["timeout"] for request in client.requests] == [10, 8]


def test_late_model_reply_does_not_authorize_edits(dispatcher, workspace, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(agent_module, "perf_counter", clock)

    def late_reply(messages, tools):
        clock.now = 6
        return tool_reply(edit_call())

    result = CodingAgent(
        ScriptedClient([late_reply]), dispatcher, limits=AgentLimits(max_seconds=5)
    ).run("Fix")
    assert result.stop_reason == "time_limit"
    assert result.tool_calls == 0
    assert "return a - b" in workspace.read_file("calculator.py")


def test_time_limit_is_checked_between_tools(dispatcher, workspace, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(agent_module, "perf_counter", clock)
    original_dispatch = dispatcher.dispatch

    def slow_dispatch(name, arguments):
        result = original_dispatch(name, arguments)
        clock.now = 6
        return result

    monkeypatch.setattr(dispatcher, "dispatch", slow_dispatch)
    client = ScriptedClient([tool_reply(read_call(), edit_call())])
    result = CodingAgent(client, dispatcher, limits=AgentLimits(max_seconds=5)).run(
        "Fix"
    )
    assert result.stop_reason == "time_limit"
    assert result.tool_calls == 1
    assert "return a - b" in workspace.read_file("calculator.py")
    assert "Not executed" in result.messages[-1].text


def test_expected_model_failure_preserves_prior_progress(dispatcher):
    client = ScriptedClient(
        [tool_reply(read_call()), ModelError("Provider unavailable")]
    )
    result = CodingAgent(client, dispatcher).run("Inspect")
    assert result.stop_reason == "model_error"
    assert result.error == "Provider unavailable"
    assert result.iterations == 2
    assert result.tool_calls == 1
    assert result.input_tokens == 1
    assert result.output_tokens == 1


def test_unexpected_client_errors_propagate(dispatcher):
    client = ScriptedClient([RuntimeError("Client bug")])
    with pytest.raises(RuntimeError, match="Client bug"):
        CodingAgent(client, dispatcher).run("Inspect")


def test_unexpected_dispatch_errors_propagate(dispatcher, monkeypatch):
    def broken_dispatch(name, arguments):
        raise RuntimeError("Dispatcher bug")

    monkeypatch.setattr(dispatcher, "dispatch", broken_dispatch)
    with pytest.raises(RuntimeError, match="Dispatcher bug"):
        CodingAgent(ScriptedClient([tool_reply(read_call())]), dispatcher).run(
            "Inspect"
        )


@pytest.mark.parametrize(
    "reply",
    [
        None,
        ModelReply(text=123),
        ModelReply(tool_calls=[]),
        ModelReply(stop_reason="unknown"),
        ModelReply(stop_reason="tool_calls"),
        ModelReply(tool_calls=(edit_call(),)),
        tool_reply(ToolCall("", "read_file", {})),
        tool_reply(ToolCall("id", "", {})),
        tool_reply("not a tool call"),
    ],
)
def test_malformed_replies_stop_before_tool_execution(dispatcher, workspace, reply):
    result = CodingAgent(ScriptedClient([reply]), dispatcher).run("Fix")
    assert result.stop_reason == "invalid_response"
    assert result.error
    assert result.tool_calls == 0
    assert "return a - b" in workspace.read_file("calculator.py")


def test_duplicate_ids_in_a_batch_prevent_all_edits(dispatcher, workspace):
    reply = tool_reply(edit_call("duplicate"), read_call("duplicate"))
    result = CodingAgent(ScriptedClient([reply]), dispatcher).run("Fix")
    assert result.stop_reason == "invalid_response"
    assert result.tool_calls == 0
    assert "return a - b" in workspace.read_file("calculator.py")


def test_ids_cannot_be_reused_across_replies(dispatcher, workspace):
    client = ScriptedClient(
        [tool_reply(read_call("same")), tool_reply(edit_call("same"))]
    )
    result = CodingAgent(client, dispatcher).run("Fix")
    assert result.stop_reason == "invalid_response"
    assert result.tool_calls == 1
    assert "return a - b" in workspace.read_file("calculator.py")


@pytest.mark.parametrize("field", ["input_tokens", "output_tokens"])
@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_invalid_token_usage_is_rejected(dispatcher, field, value):
    reply = ModelReply(**{field: value})
    result = CodingAgent(ScriptedClient([reply]), dispatcher).run("Inspect")
    assert result.stop_reason == "invalid_response"
    assert result.tool_calls == 0


def test_client_cannot_mutate_the_stored_history(dispatcher):
    def first_reply(messages, tools):
        messages.clear()
        tools.clear()
        return tool_reply(read_call())

    client = ScriptedClient([first_reply, ModelReply(text="Done")])
    result = CodingAgent(client, dispatcher).run("Inspect")
    assert result.stop_reason == "completed"
    assert client.requests[1]["messages"][0].role == "system"
    assert len(client.requests[1]["tools"]) == 4


def test_provider_content_is_preserved_without_interpretation(dispatcher):
    metadata = {"content": [{"type": "provider-block", "signature": "original"}]}

    def next_reply(messages, tools):
        metadata["content"][0]["signature"] = "changed outside the loop"
        return ModelReply(text="Done")

    client = ScriptedClient(
        [
            ModelReply(
                stop_reason="tool_calls",
                tool_calls=(read_call(),),
                provider_data=metadata,
            ),
            next_reply,
        ]
    )
    result = CodingAgent(client, dispatcher).run("Inspect")
    assert result.messages[2].provider_data["content"][0]["signature"] == "original"


def test_reusing_an_agent_starts_a_fresh_conversation(dispatcher):
    client = ScriptedClient(
        [
            tool_reply(read_call()),
            ModelReply(text="First"),
            tool_reply(read_call()),
            ModelReply(text="Second"),
        ]
    )
    agent = CodingAgent(client, dispatcher)
    first = agent.run("First task")
    second = agent.run("Second task")
    assert first.stop_reason == second.stop_reason == "completed"
    assert first.iterations == second.iterations == 2
    assert first.tool_calls == second.tool_calls == 1
    assert len(client.requests[2]["messages"]) == 2
    assert client.requests[2]["messages"][1].text == "Second task"


@pytest.mark.parametrize("task", [None, "", "  ", 123])
def test_invalid_task_is_rejected_without_a_request(dispatcher, task):
    client = ScriptedClient([])
    with pytest.raises(ValueError):
        CodingAgent(client, dispatcher).run(task)
    assert client.requests == []


@pytest.mark.parametrize(
    "field",
    ["max_iterations", "max_tool_calls", "max_output_tokens", "max_total_tokens"],
)
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_integer_limits(field, value):
    with pytest.raises(ValueError):
        AgentLimits(**{field: value})


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), True])
def test_invalid_time_limits(value):
    with pytest.raises(ValueError):
        AgentLimits(max_seconds=value)
