from __future__ import annotations

import hashlib
import importlib
import re
import shutil
import time
import zipfile
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterator, TypedDict

from .config import AppConfig
from .memory_manager import SphereMemoryManager
from .memory_writer import MemoryWriter
from .models import MemoryNode, now_iso
from .storage import Storage
from .utils import stable_content_hash, tokenize
from .vector_store import VectorStore
from .writeback import MemoryWritebackService


class _TextHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data and data.strip():
            self.parts.append(data.strip())


def _html_to_text(html: str) -> str:
    parser = _TextHTMLParser()
    parser.feed(html)
    return "\n".join(parser.parts)


def _zip_xml_text(path: Path, members: list[str]) -> str:
    parts: list[str] = []
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        for member in members:
            if member not in names:
                continue
            try:
                root = ET.fromstring(zf.read(member))
            except Exception:
                continue
            for elem in root.iter():
                if elem.text and elem.text.strip():
                    parts.append(elem.text.strip())
    return "\n".join(parts)


def _docx_to_text(path: Path) -> str:
    return _zip_xml_text(path, ["word/document.xml"])


def _pptx_to_text(path: Path) -> str:
    with zipfile.ZipFile(path) as zf:
        slide_members = sorted(name for name in zf.namelist() if name.startswith("ppt/slides/slide") and name.endswith(".xml"))
    return _zip_xml_text(path, slide_members)


def _xlsx_to_text(path: Path) -> str:
    with zipfile.ZipFile(path) as zf:
        members = zf.namelist()
        sheet_members = sorted(name for name in members if name.startswith("xl/worksheets/sheet") and name.endswith(".xml"))
        shared = []
        if "xl/sharedStrings.xml" in members:
            try:
                root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
                for elem in root.iter():
                    if elem.text and elem.text.strip():
                        shared.append(elem.text.strip())
            except Exception:
                pass
    return "\n".join(shared) + "\n" + _zip_xml_text(path, sheet_members)


class FileFingerprint(TypedDict):
    file_hash: str
    size_bytes: int
    modified_at: float


def _load_pdf_reader() -> Any:
    try:
        pypdf_module = importlib.import_module("pypdf")
    except ModuleNotFoundError as exc:
        raise RuntimeError("pypdf is required to ingest PDF files") from exc
    pdf_reader = getattr(pypdf_module, "PdfReader", None)
    if pdf_reader is None:
        raise RuntimeError("pypdf.PdfReader is not available")
    return pdf_reader


@dataclass
class IngestResult:
    path: str
    node_id: str
    chunk_count: int
    detected_kind: str
    zone: str
    cell: str
    status: str = "created"


class FileIngestor:
    def __init__(
        self,
        config: AppConfig,
        storage: Storage,
        manager: SphereMemoryManager,
        writer: MemoryWriter,
        vector_store: VectorStore,
        writeback: MemoryWritebackService | None = None,
    ) -> None:
        self.config = config
        self.storage = storage
        self.manager = manager
        self.writer = writer
        self.vector_store = vector_store
        self.writeback = writeback

    def ingest_path(
        self,
        target: str | Path,
        shell: int,
        sector: str,
        zone: str,
        recursive: bool = True,
        stage: str = "staging",
        tags: str = "",
        incremental: bool = False,
    ) -> list[IngestResult]:
        path = Path(target)
        files = self._collect_files(path, recursive=recursive)
        if self.writeback is not None and len(files) > 1:
            return self._ingest_path_batch(files, shell=shell, sector=sector, zone=zone, stage=stage, tags=tags, incremental=incremental)
        results: list[IngestResult] = []
        for file_path in files:
            result = self.ingest_file(file_path, shell=shell, sector=sector, zone=zone, stage=stage, tags=tags, incremental=incremental)
            if result is not None:
                results.append(result)
        return results

    def _ingest_path_batch(
        self,
        files: list[Path],
        shell: int,
        sector: str,
        zone: str,
        stage: str,
        tags: str,
        incremental: bool,
    ) -> list[IngestResult]:
        """Batch-ingest multiple files: prepare all nodes first, then embed all at once."""
        assert self.writeback is not None
        batch_items: list[tuple[MemoryNode, str, str, Path, FileFingerprint, str | None]] = []

        for file_path in files:
            if not file_path.exists() or not file_path.is_file():
                continue
            if file_path.suffix.lower() not in self._supported_suffixes():
                continue
            fingerprint = self._fingerprint(file_path)
            prev = self.storage.fetch_ingest_file(str(file_path.resolve())) if incremental else None
            if prev and prev.get("file_hash") == fingerprint["file_hash"]:
                continue
            parsed = self._read_file(file_path)
            if not parsed["text"].strip():
                continue
            raw_copy = self._store_raw_copy(file_path)
            cell = self._derive_cell(file_path, parsed["kind"])
            summary = self._build_summary(file_path, parsed["text"], parsed["kind"])
            old_node_id = str(prev.get("node_id")) if prev and prev.get("node_id") else None
            node = MemoryNode(
                shell=shell,
                sector=sector,
                zone=zone,
                cell=cell,
                molecular_type=parsed["kind"],
                summary=summary,
                content_hash=stable_content_hash(parsed["text"]),
                raw_content=parsed["text"],
                content_ref=str(raw_copy),
                importance=0.55 if parsed["kind"] in {"pdf", "markdown"} else 0.45,
                creative_score=0.25,
                stability_score=0.55,
                compression_level="minimal" if shell >= 4 else "medium",
                stage=stage,
                tags=tags or self._auto_tags(file_path, parsed["text"], parsed["kind"]),
            )
            batch_items.append((node, parsed["kind"], str(file_path.resolve()), file_path, fingerprint, old_node_id))

        if not batch_items:
            return []

        # Handle replacements first
        for node, kind, source_path, file_path, fingerprint, old_node_id in batch_items:
            if old_node_id:
                old_chunk_ids = self.storage.delete_chunks_for_node(old_node_id)
                old_object_ids = self.storage.delete_objects_for_node(old_node_id)
                self.storage.delete_chunk_neighbors(old_chunk_ids)
                self.vector_store.delete_chunks(old_chunk_ids)
                self.vector_store.delete_objects(old_object_ids)
                self.storage.delete_node(old_node_id)

        # Batch writeback (single embedding pass)
        node_tuples = [(node, kind, source_path) for node, kind, source_path, _, _, _ in batch_items]
        reports = self.writeback.writeback_batch(node_tuples)

        # Record ingest state
        results: list[IngestResult] = []
        for (node, kind, source_path, file_path, fingerprint, old_node_id), report in zip(batch_items, reports):
            self.storage.upsert_ingest_file(
                source_path=source_path,
                file_hash=fingerprint["file_hash"],
                size_bytes=fingerprint["size_bytes"],
                modified_at=fingerprint["modified_at"],
                node_id=node.id,
                last_ingested_at=now_iso(),
                status="active",
            )
            results.append(IngestResult(
                path=str(file_path),
                node_id=node.id,
                chunk_count=int(report["chunk_count"]),
                detected_kind=kind,
                zone=node.zone,
                cell=node.cell,
                status="updated" if old_node_id else "created",
            ))
        return results

    def ingest_file(
        self,
        file_path: str | Path,
        shell: int,
        sector: str,
        zone: str,
        stage: str = "staging",
        tags: str = "",
        incremental: bool = False,
    ) -> IngestResult | None:
        file_path = Path(file_path)
        if not file_path.exists() or not file_path.is_file():
            return None
        if file_path.suffix.lower() not in self._supported_suffixes():
            return None

        fingerprint = self._fingerprint(file_path)
        prev = self.storage.fetch_ingest_file(str(file_path.resolve())) if incremental else None
        if prev and prev.get("file_hash") == fingerprint["file_hash"]:
            return None

        parsed = self._read_file(file_path)
        if not parsed["text"].strip():
            return None
        raw_copy = self._store_raw_copy(file_path)
        cell = self._derive_cell(file_path, parsed["kind"])
        summary = self._build_summary(file_path, parsed["text"], parsed["kind"])

        status = "updated" if prev else "created"
        old_node_id = str(prev.get("node_id")) if prev and prev.get("node_id") else None

        node = MemoryNode(
            shell=shell,
            sector=sector,
            zone=zone,
            cell=cell,
            molecular_type=parsed["kind"],
            summary=summary,
            content_hash=stable_content_hash(parsed["text"]),
            raw_content=parsed["text"],
            content_ref=str(raw_copy),
            importance=0.55 if parsed["kind"] in {"pdf", "markdown"} else 0.45,
            creative_score=0.25,
            stability_score=0.55,
            compression_level="minimal" if shell >= 4 else "medium",
            stage=stage,
            tags=tags or self._auto_tags(file_path, parsed["text"], parsed["kind"]),
        )
        report = (
            self.writeback.writeback_memory(
                node=node,
                source_kind=parsed["kind"],
                source_path=str(file_path.resolve()),
                replace_node_id=old_node_id,
            )
            if self.writeback is not None
            else self._legacy_writeback(node, parsed["kind"], str(file_path.resolve()), old_node_id)
        )
        self.storage.upsert_ingest_file(
            source_path=str(file_path.resolve()),
            file_hash=fingerprint["file_hash"],
            size_bytes=fingerprint["size_bytes"],
            modified_at=fingerprint["modified_at"],
            node_id=node.id,
            last_ingested_at=now_iso(),
            status="active",
        )
        return IngestResult(
            path=str(file_path),
            node_id=node.id,
            chunk_count=int(report["chunk_count"]),
            detected_kind=parsed["kind"],
            zone=zone,
            cell=cell,
            status=status,
        )

    def sync_path(
        self,
        target: str | Path,
        shell: int,
        sector: str,
        zone: str,
        recursive: bool = True,
        stage: str = "staging",
        tags: str = "",
    ) -> list[IngestResult]:
        return self.ingest_path(target, shell=shell, sector=sector, zone=zone, recursive=recursive, stage=stage, tags=tags, incremental=True)

    def watch_path(
        self,
        target: str | Path,
        shell: int,
        sector: str,
        zone: str,
        recursive: bool = True,
        stage: str = "staging",
        tags: str = "",
        poll_seconds: float | None = None,
        max_rounds: int = 0,
    ) -> Iterator[list[IngestResult]]:
        rounds = 0
        poll_seconds = poll_seconds or self.config.watch_poll_seconds
        while True:
            yield self.sync_path(target, shell=shell, sector=sector, zone=zone, recursive=recursive, stage=stage, tags=tags)
            rounds += 1
            if max_rounds > 0 and rounds >= max_rounds:
                break
            time.sleep(poll_seconds)

    def _supported_suffixes(self) -> set[str]:
        return {
            ".md", ".markdown", ".txt", ".py", ".js", ".ts", ".tsx", ".jsx", ".json", ".yaml", ".yml",
            ".toml", ".ini", ".cfg", ".log", ".out", ".err", ".pdf", ".docx", ".pptx", ".xlsx",
            ".html", ".htm", ".java", ".cpp", ".c", ".rs", ".go", ".sql", ".sh", ".ps1",
        }

    def _legacy_writeback(self, node: MemoryNode, source_kind: str, source_path: str, old_node_id: str | None) -> dict[str, int | str]:
        if old_node_id:
            old_chunk_ids = self.storage.delete_chunks_for_node(old_node_id)
            old_object_ids = self.storage.delete_objects_for_node(old_node_id)
            self.storage.delete_chunk_neighbors(old_chunk_ids)
            self.vector_store.delete_chunks(old_chunk_ids)
            self.vector_store.delete_objects(old_object_ids)
            self.storage.delete_node(old_node_id)
        self.manager.add_node(node)
        chunks = self.writer.prepare_chunks(node, source_kind=source_kind, source_path=source_path)
        self.storage.insert_chunks(chunks)
        neighbors = self.writer.build_chunk_neighbors(chunks)
        self.storage.insert_chunk_neighbors(neighbors)
        self.vector_store.upsert_chunks(chunks)
        objects = self.writer.extract_objects(node, chunks)
        self.storage.insert_objects(objects)
        self.vector_store.upsert_objects(objects)
        edges = self.writer.create_edges_for_new_node(node)
        for edge in edges:
            self.storage.insert_edge(asdict(edge))
        return {
            "node_id": node.id,
            "chunk_count": len(chunks),
            "object_count": len(objects),
            "neighbor_count": len(neighbors),
            "edge_count": len(edges),
        }

    def _collect_files(self, path: Path, recursive: bool) -> list[Path]:
        if path.is_file():
            return [path]
        pattern = "**/*" if recursive else "*"
        candidates = [p for p in path.glob(pattern) if p.is_file()]
        return [p for p in candidates if p.suffix.lower() in self._supported_suffixes()]

    def _read_file(self, path: Path) -> dict[str, str]:
        suffix = path.suffix.lower()
        if suffix in {".md", ".markdown"}:
            return {"kind": "markdown", "text": path.read_text(encoding="utf-8", errors="ignore")}
        if suffix in {".log", ".out", ".err"}:
            return {"kind": "log", "text": path.read_text(encoding="utf-8", errors="ignore")}
        if suffix == ".pdf":
            pdf_reader = _load_pdf_reader()
            reader = pdf_reader(str(path))
            texts = []
            for page in reader.pages:
                texts.append(page.extract_text() or "")
            return {"kind": "pdf", "text": "\n\n".join(texts)}
        if suffix in {".html", ".htm"}:
            return {"kind": "html", "text": _html_to_text(path.read_text(encoding="utf-8", errors="ignore"))}
        if suffix == ".docx":
            return {"kind": "docx", "text": _docx_to_text(path)}
        if suffix == ".pptx":
            return {"kind": "pptx", "text": _pptx_to_text(path)}
        if suffix == ".xlsx":
            return {"kind": "xlsx", "text": _xlsx_to_text(path)}
        if suffix in {".txt", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg"}:
            return {"kind": "document", "text": path.read_text(encoding="utf-8", errors="ignore")}
        return {"kind": "code", "text": path.read_text(encoding="utf-8", errors="ignore")}

    def _build_summary(self, path: Path, text: str, kind: str) -> str:
        cleaned = re.sub(r"\s+", " ", text).strip()
        preview = cleaned[:240]
        if kind == "code":
            return f"Code file {path.name}: {preview}"
        if kind == "log":
            return f"Log file {path.name}: {preview}"
        if kind == "pdf":
            return f"PDF document {path.name}: {preview}"
        if kind in {"docx", "pptx", "xlsx", "html"}:
            return f"{kind.upper()} document {path.name}: {preview}"
        if kind == "markdown":
            return f"Markdown file {path.name}: {preview}"
        return f"Document {path.name}: {preview}"

    def _derive_cell(self, path: Path, kind: str) -> str:
        stem = re.sub(r"[^a-zA-Z0-9_]+", "_", path.stem.lower()).strip("_") or "file"
        return f"{kind}_{stem}"

    def _auto_tags(self, path: Path, text: str, kind: str) -> str:
        base = {kind, path.suffix.lower().lstrip(".")}
        toks = [t for t in tokenize(path.name + " " + text[:500]) if len(t) > 4]
        base.update(toks[:8])
        return ",".join(sorted(base))

    def _store_raw_copy(self, path: Path) -> Path:
        digest = hashlib.md5(str(path.resolve()).encode("utf-8")).hexdigest()[:10]
        dest = self.config.raw_dir / f"{digest}_{path.name}"
        shutil.copy2(path, dest)
        return dest

    def _fingerprint(self, path: Path) -> FileFingerprint:
        stat = path.stat()
        h = hashlib.sha1()
        with path.open("rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
        return {
            "file_hash": h.hexdigest(),
            "size_bytes": int(stat.st_size),
            "modified_at": float(stat.st_mtime),
        }
