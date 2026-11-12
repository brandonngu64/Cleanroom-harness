"""Persist observations from a run without treating model completion as proof."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from .agent import AgentResult
from .runner import RunResult


@dataclass(frozen=True)
class RunReport:
    task: str
    repository: str | Path
    model: str
    agent: AgentResult | None = None
    baseline: RunResult | None = None
    verification: RunResult | None = None
    source_revision: str | None = None
    patch: str = ""
    error: str | None = None
    duration_seconds: float | None = None
    configuration: dict[str, object] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self):
        for name in ("task", "model"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be nonempty text")
        if not isinstance(self.repository, (str, Path)) or not str(self.repository):
            raise ValueError("repository must be a nonempty path")
        if not isinstance(self.patch, str):
            raise ValueError("patch must be text")
        if self.duration_seconds is not None and (
            not isinstance(self.duration_seconds, (int, float))
            or isinstance(self.duration_seconds, bool)
            or not math.isfinite(self.duration_seconds)
            or self.duration_seconds < 0
        ):
            raise ValueError("duration_seconds must be finite and nonnegative")


def command_status(result: RunResult | None) -> str:
    """Describe the configured command's outcome, not whether a task was solved."""
    if result is None:
        return "not_run"
    if result.error is not None:
        return "error"
    if result.timed_out:
        return "timed_out"
    if result.exit_code is None:
        return "unverified"
    return "passed" if result.succeeded else "failed"


def _path_to_string(value: object) -> str:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported report value: {type(value).__name__}")


def save_report(report: RunReport, output_directory: str | Path) -> Path:
    """Write three UTF-8 artifacts into a new, exclusively created directory.

    Serialization is checked before creating the directory. Existing output is
    never overwritten. I/O failures propagate and may leave partial artifacts;
    report.json is written last. Git patches, baseline tests, and verification
    are supplied by the caller; this function does not perform those operations.
    """
    summary = asdict(report)
    transcript = []
    if summary["agent"] is not None:
        transcript = summary["agent"].pop("messages")
    patch = summary.pop("patch")
    summary.update(
        {
            "schema_version": 1,
            "baseline_status": command_status(report.baseline),
            "verification_status": command_status(report.verification),
            "transcript_file": "transcript.json",
            "patch_file": "changes.patch",
        }
    )

    def encode_json(value: object) -> bytes:
        text = json.dumps(
            value,
            default=_path_to_string,
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
        )
        return (text + "\n").encode("utf-8")

    artifacts = {
        "transcript.json": encode_json(transcript),
        "changes.patch": patch.encode("utf-8"),
        "report.json": encode_json(summary),
    }
    target = Path(output_directory).resolve()
    target.mkdir(parents=True, exist_ok=False)
    for name, data in artifacts.items():
        with (target / name).open("xb") as stream:
            stream.write(data)
    return target
