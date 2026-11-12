from pathlib import Path

import pytest

from harness.workspace import Workspace


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    """Give each test a workspace inside its own temporary directory."""
    root = tmp_path / "repo"
    root.mkdir()
    return Workspace(root)


@pytest.mark.parametrize("existing_file", [False, True])
def test_invalid_workspace_root(tmp_path, existing_file):
    root = tmp_path / "invalid"

    if existing_file:
        root.write_text("Not a directory", encoding="utf-8")

    with pytest.raises(ValueError):
        Workspace(root)


def test_read_file(workspace):
    target = workspace.root / "notes.txt"
    target.write_text("Hello, caf\u00e9!\n", encoding="utf-8")

    assert workspace.read_file("notes.txt") == "Hello, caf\u00e9!\n"


def test_read_missing_file(workspace):
    with pytest.raises(FileNotFoundError):
        workspace.read_file("missing.txt")


def test_list_files(workspace):
    (workspace.root / "README.md").write_text(
        "Example project\n", encoding="utf-8"
    )

    package = workspace.root / "pkg"
    package.mkdir()
    (package / "module.py").write_text("x = 1\n", encoding="utf-8")

    (workspace.root / "empty_directory").mkdir()

    git = workspace.root / ".git"
    git.mkdir()
    (git / "config").write_text("metadata\n", encoding="utf-8")

    expected = ["README.md", str(Path("pkg") / "module.py")]

    assert workspace.list_files() == sorted(expected)


def test_replace_text(workspace):
    target = workspace.root / "calculator.py"
    target.write_text(
        "def add(a, b):\n    return a - b\n",
        encoding="utf-8",
    )

    workspace.replace_text(
        "calculator.py",
        "return a - b",
        "return a + b",
    )

    assert target.read_text(encoding="utf-8") == (
        "def add(a, b):\n    return a + b\n"
    )


@pytest.mark.parametrize(
    "old",
    ["", "subtract", "add"],
    ids=["empty-search", "missing-match", "multiple-matches"],
)
def test_invalid_replacement_leaves_file_unchanged(workspace, old):
    target = workspace.root / "notes.txt"
    original = "add add\n"
    target.write_text(original, encoding="utf-8")

    with pytest.raises(ValueError):
        workspace.replace_text("notes.txt", old, "replacement")

    assert target.read_text(encoding="utf-8") == original


@pytest.mark.parametrize("absolute", [False, True])
def test_outside_paths_are_rejected(workspace, tmp_path, absolute):
    outside = tmp_path / "outside.txt"
    outside.write_text("original\n", encoding="utf-8")

    if absolute:
        path = str(outside.resolve())
    else:
        path = "../outside.txt"

    with pytest.raises(ValueError):
        workspace.read_file(path)

    with pytest.raises(ValueError):
        workspace.replace_text(path, "original", "changed")

    assert outside.read_text(encoding="utf-8") == "original\n"


def test_git_metadata_is_protected(workspace):
    git = workspace.root / ".git"
    git.mkdir()

    config = git / "config"
    config.write_text("original\n", encoding="utf-8")

    with pytest.raises(ValueError):
        workspace.read_file(".git/config")

    with pytest.raises(ValueError):
        workspace.replace_text(".git/config", "original", "changed")

    assert config.read_text(encoding="utf-8") == "original\n"
