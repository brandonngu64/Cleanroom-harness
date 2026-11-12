from dataclasses import asdict
import difflib
import json
import sys

import pytest

anthropic = pytest.importorskip("anthropic")
httpx = pytest.importorskip("httpx2")

from harness.agent import CodingAgent, Message, ModelError, ToolCall
from harness.claude import ClaudeClient
from harness.report import RunReport, save_report
from harness.runner import LocalRunner
from harness.tools import ToolDispatcher, get_tool_definitions
from harness.workspace import Workspace


def response_body(content=None, stop_reason="end_turn", usage=None):
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "test-model",
        "content": (
            content if content is not None else [{"type": "text", "text": "Done"}]
        ),
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": (
            usage if usage is not None else {"input_tokens": 3, "output_tokens": 2}
        ),
    }


def sdk_client(handler):
    return anthropic.Anthropic(
        api_key="test-key",
        max_retries=5,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def initial_messages():
    return [Message("system", "System instructions"), Message("user", "Fix the code")]


def test_real_sdk_request_translation_and_reply_usage():
    captured = []

    def handler(request):
        captured.append(request)
        return httpx.Response(
            200,
            json=response_body(
                usage={
                    "input_tokens": 5,
                    "output_tokens": 3,
                    "cache_creation_input_tokens": 7,
                    "cache_read_input_tokens": 11,
                }
            ),
        )

    with sdk_client(handler) as sdk:
        adapter = ClaudeClient("test-model", client=sdk)
        reply = adapter.complete(
            initial_messages(), get_tool_definitions(), max_output_tokens=128, timeout=3
        )

    payload = json.loads(captured[0].content)
    assert payload["model"] == "test-model"
    assert payload["system"] == "System instructions"
    assert payload["max_tokens"] == 128
    assert payload["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "Fix the code"}]}
    ]
    assert payload["tools"][0]["input_schema"]["type"] == "object"
    assert "parameters" not in payload["tools"][0]
    assert captured[0].extensions["timeout"]["read"] == 3
    assert reply.stop_reason == "completed"
    assert reply.text == "Done"
    assert reply.input_tokens == 23
    assert reply.output_tokens == 3
    assert reply.provider_data["provider"] == "anthropic"
    assert reply.provider_data["response"]["id"] == "msg_test"


def test_tool_batch_results_and_original_thinking_content_are_preserved():
    captured = []
    original = [
        {
            "type": "thinking",
            "thinking": "Visible reasoning",
            "signature": "signed-content",
        },
        {
            "type": "tool_use",
            "id": "read-1",
            "name": "read_file",
            "input": {"path": "a.py"},
        },
        {"type": "tool_use", "id": "list-1", "name": "list_files", "input": {}},
    ]

    def handler(request):
        captured.append(json.loads(request.content))
        body = (
            response_body(original, "tool_use")
            if len(captured) == 1
            else response_body()
        )
        return httpx.Response(200, json=body)

    with sdk_client(handler) as sdk:
        adapter = ClaudeClient("test-model", client=sdk)
        first = adapter.complete(
            initial_messages(), get_tool_definitions(), max_output_tokens=128, timeout=3
        )
        messages = initial_messages() + [
            Message(
                "assistant",
                text=first.text,
                tool_calls=first.tool_calls,
                provider_data=first.provider_data,
            ),
            Message("tool", text='{"content": "source"}', tool_call_id="read-1"),
            Message(
                "tool",
                text='{"error": "example"}',
                tool_call_id="list-1",
                is_error=True,
            ),
        ]
        adapter.complete(
            messages, get_tool_definitions(), max_output_tokens=128, timeout=3
        )

    sent = captured[1]["messages"]
    assert sent[1]["content"] == original
    assert sent[2]["role"] == "user"
    assert [block["tool_use_id"] for block in sent[2]["content"]] == [
        "read-1",
        "list-1",
    ]
    assert [block["is_error"] for block in sent[2]["content"]] == [False, True]
    assert len(sent) == 3


def test_generic_assistant_message_can_be_translated_without_provider_metadata():
    captured = []

    def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=response_body())

    messages = initial_messages() + [
        Message(
            "assistant",
            text="Reading",
            tool_calls=(ToolCall("read-1", "read_file", {"path": "a.py"}),),
        ),
        Message("tool", text="contents", tool_call_id="read-1"),
    ]
    with sdk_client(handler) as sdk:
        ClaudeClient("test-model", client=sdk).complete(
            messages, get_tool_definitions(), max_output_tokens=128, timeout=3
        )
    assert captured[0]["messages"][1]["content"] == [
        {"type": "text", "text": "Reading"},
        {
            "type": "tool_use",
            "id": "read-1",
            "name": "read_file",
            "input": {"path": "a.py"},
        },
    ]


@pytest.mark.parametrize(
    "native, normalized",
    [
        ("end_turn", "completed"),
        ("tool_use", "tool_calls"),
        ("max_tokens", "output_limit"),
        ("model_context_window_exceeded", "output_limit"),
        ("refusal", "refused"),
    ],
)
def test_stop_reasons_are_normalized(native, normalized):
    content = (
        [{"type": "tool_use", "id": "list-1", "name": "list_files", "input": {}}]
        if native == "tool_use"
        else None
    )
    with sdk_client(
        lambda request: httpx.Response(200, json=response_body(content, native))
    ) as sdk:
        reply = ClaudeClient("test-model", client=sdk).complete(
            initial_messages(), [], max_output_tokens=128, timeout=3
        )
    assert reply.stop_reason == normalized
    assert reply.provider_data["response"]["stop_reason"] == native


@pytest.mark.parametrize("reason", ["pause_turn", "stop_sequence", "unexpected"])
def test_unsupported_stop_reasons_fail_explicitly(reason):
    with sdk_client(
        lambda request: httpx.Response(200, json=response_body(stop_reason=reason))
    ) as sdk:
        with pytest.raises(ModelError, match="stop reason"):
            ClaudeClient("test-model", client=sdk).complete(
                initial_messages(), [], max_output_tokens=128, timeout=3
            )


@pytest.mark.parametrize("status", [401, 429, 500])
def test_api_failures_are_normalized_without_automatic_retries(status):
    attempts = []

    def handler(request):
        attempts.append(request)
        return httpx.Response(
            status,
            json={
                "type": "error",
                "error": {"type": "api_error", "message": "Simulated failure"},
            },
        )

    with sdk_client(handler) as sdk:
        with pytest.raises(ModelError) as exc_info:
            ClaudeClient("test-model", client=sdk).complete(
                initial_messages(), [], max_output_tokens=128, timeout=3
            )
    assert isinstance(exc_info.value.__cause__, anthropic.APIError)
    assert len(attempts) == 1


def test_transport_timeout_becomes_a_model_error():
    attempts = []

    def handler(request):
        attempts.append(request)
        raise httpx.ReadTimeout("Simulated timeout", request=request)

    with sdk_client(handler) as sdk:
        with pytest.raises(ModelError) as exc_info:
            ClaudeClient("test-model", client=sdk).complete(
                initial_messages(), [], max_output_tokens=128, timeout=3
            )
    assert isinstance(exc_info.value.__cause__, anthropic.APITimeoutError)
    assert len(attempts) == 1


@pytest.mark.parametrize(
    "messages",
    [
        [Message("tool", text="unmatched", tool_call_id="no-call")],
        initial_messages()
        + [Message("assistant", tool_calls=(ToolCall("id", "read_file", {}),))],
        initial_messages()
        + [
            Message("assistant", tool_calls=(ToolCall("id", "read_file", {}),)),
            Message("user", "intervening"),
        ],
        [Message("assistant", text="wrong first role")],
        initial_messages() + [Message("system", "late system")],
    ],
)
def test_invalid_history_is_rejected_before_a_request(messages):
    def forbidden(request):
        pytest.fail("Invalid history reached the API")

    with sdk_client(forbidden) as sdk:
        with pytest.raises(ValueError):
            ClaudeClient("test-model", client=sdk).complete(
                messages, [], max_output_tokens=128, timeout=3
            )


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_output_allowance_is_rejected(value):
    with sdk_client(lambda request: pytest.fail("Unexpected request")) as sdk:
        with pytest.raises(ValueError):
            ClaudeClient("test-model", client=sdk).complete(
                initial_messages(), [], max_output_tokens=value, timeout=3
            )


@pytest.mark.parametrize("value", [0, -1, True, float("inf"), float("nan")])
def test_invalid_timeout_is_rejected(value):
    with sdk_client(lambda request: pytest.fail("Unexpected request")) as sdk:
        with pytest.raises(ValueError):
            ClaudeClient("test-model", client=sdk).complete(
                initial_messages(), [], max_output_tokens=128, timeout=value
            )


def test_adapter_does_not_close_an_injected_client():
    with sdk_client(lambda request: httpx.Response(200, json=response_body())) as sdk:
        with ClaudeClient("test-model", client=sdk):
            pass
        assert not sdk.is_closed()


def test_default_client_is_closed_and_uses_environment_credentials(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with ClaudeClient("test-model") as adapter:
        assert adapter._client.max_retries == 0
        assert not adapter._client.is_closed()
    assert adapter._client.is_closed()


@pytest.mark.parametrize("model", ["", "  ", None])
def test_model_id_is_required(model):
    with pytest.raises(ValueError):
        ClaudeClient(model)


def test_http_simulated_model_workflow_saves_verified_results(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    original = "def add(a, b):\n    return a - b\n"
    (root / "calculator.py").write_text(original, encoding="utf-8")
    (root / "check.py").write_text(
        "from calculator import add\nassert add(2, 3) == 5\n", encoding="utf-8"
    )
    runner = LocalRunner()
    command = [sys.executable, "-B", "check.py"]
    baseline = runner.run(command, cwd=root)
    requests = []
    replies = [
        response_body(
            [
                {
                    "type": "tool_use",
                    "id": "read-1",
                    "name": "read_file",
                    "input": {"path": "calculator.py"},
                }
            ],
            "tool_use",
        ),
        response_body(
            [
                {
                    "type": "tool_use",
                    "id": "edit-1",
                    "name": "replace_text",
                    "input": {
                        "path": "calculator.py",
                        "old": "return a - b",
                        "new": "return a + b",
                    },
                }
            ],
            "tool_use",
        ),
        response_body(
            [{"type": "tool_use", "id": "tests-1", "name": "run_tests", "input": {}}],
            "tool_use",
        ),
        response_body([{"type": "text", "text": "Fixed addition; check passed."}]),
    ]

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=replies[len(requests) - 1])

    dispatcher = ToolDispatcher(
        Workspace(root), runner, test_command=command, editable_files=["calculator.py"]
    )
    with sdk_client(handler) as sdk:
        agent = CodingAgent(ClaudeClient("test-model", client=sdk), dispatcher)
        outcome = agent.run("Fix addition")
    verification = runner.run(command, cwd=root)
    updated = (root / "calculator.py").read_text(encoding="utf-8")
    patch = "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            updated.splitlines(keepends=True),
            fromfile="a/calculator.py",
            tofile="b/calculator.py",
        )
    )
    report = RunReport(
        task="Fix addition",
        repository=root,
        model="test-model",
        agent=outcome,
        baseline=baseline,
        verification=verification,
        patch=patch,
    )
    saved = save_report(report, tmp_path / "artifacts")

    summary = json.loads((saved / "report.json").read_text(encoding="utf-8"))
    transcript = json.loads((saved / "transcript.json").read_text(encoding="utf-8"))
    assert summary["baseline_status"] == "failed"
    assert summary["verification_status"] == "passed"
    assert summary["agent"]["stop_reason"] == "completed"
    assert len(requests) == 4
    assert transcript[-1]["text"] == "Fixed addition; check passed."
    assert "return a + b" in (saved / "changes.patch").read_text(encoding="utf-8")
    json.dumps(asdict(outcome))
