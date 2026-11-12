
from pathlib import Path


class Workspace:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

        if not self.root.is_dir():
            raise ValueError(f"Invalid workspace: {root}")

    def _resolve_path(self, path: str) -> Path:
        target = (self.root / path).resolve()

        if not target.is_relative_to(self.root):
            raise ValueError("Path is outside the workspace")

        if ".git" in target.relative_to(self.root).parts:
            raise ValueError("Cannot access Git metadata")

        return target

    def list_files(self) -> list[str]:
        files = []

        for path in self.root.rglob("*"):
            if not path.is_file():
                continue

            try:
                resolved = self._resolve_path(
                    str(path.relative_to(self.root))
                )
            except ValueError:
                continue

            files.append(str(resolved.relative_to(self.root)))

        return sorted(files)

    def read_file(self, path: str) -> str:
        target = self._resolve_path(path)

        return target.read_text(encoding="utf-8")

    def replace_text(
        self,
        path: str,
        old: str,
        new: str
    ) -> None:
        target = self._resolve_path(path)
        content = target.read_text(encoding="utf-8")

        if not old:
            raise ValueError("Old text cannot be empty")

        if content.count(old) != 1:
            raise ValueError(
                "Expected exactly one matching occurrence"
            )

        updated = content.replace(old, new, 1)
        target.write_text(updated, encoding="utf-8")