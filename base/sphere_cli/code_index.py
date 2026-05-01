from __future__ import annotations

import ast
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .utils import stable_content_hash


@dataclass
class CodeSymbol:
    name: str
    kind: str
    lineno: int


@dataclass
class CodeFileRecord:
    path: str
    module: str
    symbols: list[CodeSymbol] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    test_file_guess: str | None = None
    parse_error: str | None = None


def code_index_dir(base_dir: Path) -> Path:
    return base_dir / "artifacts" / "code_index"


def code_index_path(base_dir: Path, project: str) -> Path:
    slug = stable_content_hash(project)[:12]
    return code_index_dir(base_dir) / f"{project.lower()}_{slug}.json"


def _module_name(root: Path, path: Path) -> str:
    rel = path.relative_to(root)
    parts = list(rel.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _test_guess(root: Path, path: Path) -> str | None:
    rel = path.relative_to(root)
    if rel.parts and rel.parts[0] == "tests":
        return None
    stem = path.stem
    candidates = [
        root / "tests" / f"test_{stem}.py",
        root / "tests" / rel.parent / f"test_{stem}.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.relative_to(root))
    return None


def parse_python_file(root: Path, path: Path) -> CodeFileRecord:
    record = CodeFileRecord(path=str(path.relative_to(root)), module=_module_name(root, path), test_file_guess=_test_guess(root, path))
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError as exc:
        record.parse_error = f"{exc.__class__.__name__}: {exc.msg} line {exc.lineno}"
        return record
    except Exception as exc:
        record.parse_error = f"{exc.__class__.__name__}: {exc}"
        return record
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            record.symbols.append(CodeSymbol(name=node.name, kind="class", lineno=node.lineno))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            record.symbols.append(CodeSymbol(name=node.name, kind="function", lineno=node.lineno))
        elif isinstance(node, ast.Import):
            record.imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = "." * node.level + (node.module or "")
            record.imports.append(module)
    record.symbols.sort(key=lambda item: (item.lineno, item.kind, item.name))
    record.imports = sorted(set(record.imports))
    return record


def build_code_index(base_dir: Path, target: Path, *, project: str = "DysonSpherain") -> dict[str, Any]:
    root = target.resolve()
    if root.is_file():
        files = [root]
        source_root = root.parent
    else:
        source_root = root
        files = sorted(
            path
            for path in root.rglob("*.py")
            if ".venv" not in path.parts and "__pycache__" not in path.parts and ".git" not in path.parts
        )
    records = [parse_python_file(source_root, path) for path in files]
    payload = {
        "schema": "dysonspherain.code_index.v1",
        "project": project,
        "root": str(source_root),
        "file_count": len(records),
        "parse_error_count": sum(1 for record in records if record.parse_error),
        "records": [asdict(record) for record in records],
    }
    path = code_index_path(base_dir, project)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def load_code_index(base_dir: Path, project: str = "DysonSpherain") -> dict[str, Any]:
    path = code_index_path(base_dir, project)
    if not path.exists():
        return {"schema": "dysonspherain.code_index.v1", "project": project, "records": []}
    return json.loads(path.read_text(encoding="utf-8"))


def search_symbol(base_dir: Path, query: str, *, project: str = "DysonSpherain") -> list[dict[str, Any]]:
    payload = load_code_index(base_dir, project)
    q = query.lower()
    matches: list[dict[str, Any]] = []
    for record in payload.get("records", []):
        for symbol in record.get("symbols", []):
            if q in str(symbol.get("name", "")).lower():
                matches.append({"path": record.get("path"), "module": record.get("module"), **symbol})
    return matches


def relevant_files(base_dir: Path, query: str, *, project: str = "DysonSpherain", limit: int = 20) -> list[dict[str, Any]]:
    payload = load_code_index(base_dir, project)
    terms = [term.lower() for term in query.replace("_", " ").split() if term.strip()]
    scored: list[tuple[int, dict[str, Any]]] = []
    for record in payload.get("records", []):
        haystack = " ".join(
            [
                str(record.get("path", "")),
                str(record.get("module", "")),
                " ".join(str(symbol.get("name", "")) for symbol in record.get("symbols", [])),
                " ".join(str(item) for item in record.get("imports", [])),
            ]
        ).lower()
        score = sum(1 for term in terms if term in haystack)
        if score:
            scored.append((score, record))
    scored.sort(key=lambda item: (-item[0], item[1].get("path", "")))
    return [{"score": score, **record} for score, record in scored[:limit]]
