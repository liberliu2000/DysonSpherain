from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from .activation_engine import ActivationEngine
from .answer_generator import EvidenceAnswerGenerator
from .background_tasks import BackgroundTaskRunner
from .config import AppConfig
from .context_assembler import ContextAssembler
from .creative_reflection_engine import CreativeReflectionEngine
from .evidence_pipeline import EvidencePipeline
from .ingestion import FileIngestor
from .memory_auditor import MemoryAuditor
from .memory_manager import SphereMemoryManager
from .memory_writer import MemoryWriter
from .models import CognitiveAugmentationResult, EvidenceRetrievalResult, MemoryNode, StructuredCompletionResult
from .path_router import PathRouter
from .reranker import RetrievalReranker
from .storage import Storage
from .vector_store import VectorStore
from .workspace import WorkspaceContext
from .writeback import MemoryWritebackService


@dataclass
class RuntimeServices:
    config: AppConfig
    storage: Storage
    vector_store: VectorStore
    manager: SphereMemoryManager
    activation: ActivationEngine
    router: PathRouter
    assembler: ContextAssembler
    writer: MemoryWriter
    auditor: MemoryAuditor
    creative: CreativeReflectionEngine
    reranker: RetrievalReranker
    writeback: MemoryWritebackService
    ingestor: FileIngestor
    background: BackgroundTaskRunner
    answer_generator: EvidenceAnswerGenerator


class UnifiedMemoryRuntime:
    def __init__(self, services: RuntimeServices) -> None:
        self.services = services
        self.evidence = EvidencePipeline(
            services.storage,
            services.vector_store,
            services.activation,
            services.router,
            config=services.config,
            creative_engine=services.creative,
        )

    @classmethod
    def from_base_dir(cls, base_dir: Path | None = None, config_overrides: dict[str, object] | None = None) -> "UnifiedMemoryRuntime":
        config = AppConfig.from_env(base_dir=base_dir or Path.cwd(), overrides=config_overrides)
        storage = Storage(config)
        storage.init_db()
        vector_store = VectorStore(config, storage=storage)
        manager = SphereMemoryManager(storage)
        activation = ActivationEngine(storage, vector_store)
        router = PathRouter()
        assembler = ContextAssembler(config)
        writer = MemoryWriter(storage, config)
        auditor = MemoryAuditor(storage, vector_store)
        creative = CreativeReflectionEngine()
        reranker = RetrievalReranker(config)
        writeback = MemoryWritebackService(storage, vector_store, manager, writer)
        ingestor = FileIngestor(config, storage, manager, writer, vector_store, writeback=writeback)
        background = BackgroundTaskRunner(config, storage, vector_store, writer)
        answer_generator = EvidenceAnswerGenerator()
        return cls(
            RuntimeServices(
                config=config,
                storage=storage,
                vector_store=vector_store,
                manager=manager,
                activation=activation,
                router=router,
                assembler=assembler,
                writer=writer,
                auditor=auditor,
                creative=creative,
                reranker=reranker,
                writeback=writeback,
                ingestor=ingestor,
                background=background,
                answer_generator=answer_generator,
            )
        )

    def _active_route_context(self, route_context: dict[str, Any] | None = None) -> dict[str, Any]:
        context = WorkspaceContext.from_values(
            workspace=self.services.config.workspace_name,
            project=self.services.config.project_name,
            session_id=self.services.config.session_id,
            scope=self.services.config.scope,
            scope_order=self.services.config.retrieval_scope_order,
            mode=self.services.config.mode,
        ).to_route_context()
        context.update(dict(route_context or {}))
        return context

    def retrieve_evidence(
        self,
        query: str,
        task_type: str = "qa",
        top_k: int = 8,
        route_context: dict[str, Any] | None = None,
    ) -> EvidenceRetrievalResult:
        return self.evidence.retrieve_evidence(
            query=query,
            task_type=task_type,
            evidence_top_k=top_k,
            route_context=self._active_route_context(route_context),
        )

    def complete_with_objects(
        self,
        query: str,
        evidence: EvidenceRetrievalResult,
        support_top_k: int = 4,
        object_top_k: int = 4,
    ) -> StructuredCompletionResult:
        return self.evidence.complete_with_objects(
            query=query,
            evidence=evidence,
            support_top_k=support_top_k,
            object_top_k=object_top_k,
        )

    def augment_cognitively(
        self,
        query: str,
        task_type: str,
        completion: StructuredCompletionResult,
        cognitive_top_k: int = 4,
    ) -> CognitiveAugmentationResult:
        return self.evidence.augment_cognitively(
            query=query,
            task_type=task_type,
            completion=completion,
            cognitive_top_k=cognitive_top_k,
        )

    def assemble_context(
        self,
        task: str,
        task_type: str,
        temperature: float,
        completion: StructuredCompletionResult,
        cognitive: CognitiveAugmentationResult,
        max_tokens: int = 1800,
    ):
        return self.services.assembler.assemble_evidence_first_with_paths(
            task=task,
            task_type=task_type,
            temperature=temperature,
            core_evidence=completion.core_evidence,
            evidence_objects=completion.evidence_objects,
            supporting_context=completion.supporting_context,
            relevant_experience=cognitive.relevant_experience,
            creative_reflections=cognitive.creative_reflections,
            alternative_paths=cognitive.alternative_paths,
            max_tokens=max_tokens,
        )

    def run_query(
        self,
        query: str,
        task_type: str = "qa",
        temperature: float = 0.5,
        max_tokens: int = 1800,
        evidence_top_k: int = 8,
        support_top_k: int = 4,
        object_top_k: int = 4,
        cognitive_top_k: int = 4,
    ) -> dict[str, Any]:
        active_route_context = self._active_route_context()
        evidence = self.retrieve_evidence(query, task_type=task_type, top_k=evidence_top_k, route_context=active_route_context)
        completion = self.complete_with_objects(
            query=query,
            evidence=evidence,
            support_top_k=support_top_k,
            object_top_k=object_top_k,
        )
        cognitive = self.augment_cognitively(
            query=query,
            task_type=task_type,
            completion=completion,
            cognitive_top_k=cognitive_top_k,
        )
        assemble_start = perf_counter()
        bundle = self.assemble_context(
            task=query,
            task_type=task_type,
            temperature=temperature,
            completion=completion,
            cognitive=cognitive,
            max_tokens=max_tokens,
        )
        bundle.debug["assemble_ms"] = round((perf_counter() - assemble_start) * 1000.0, 2)
        return {
            "evidence": evidence,
            "completion": completion,
            "cognitive": cognitive,
            "bundle": bundle,
            "alternative_paths": cognitive.alternative_paths,
        }

    def answer(
        self,
        query: str,
        task_type: str = "qa",
        temperature: float = 0.5,
        max_tokens: int = 1800,
        evidence_top_k: int = 8,
        support_top_k: int = 4,
        object_top_k: int = 4,
        cognitive_top_k: int = 4,
        answer_mode: str = "local",
        include_creative: bool = False,
    ) -> dict[str, Any]:
        run_result = self.run_query(
            query=query,
            task_type=task_type,
            temperature=temperature,
            max_tokens=max_tokens,
            evidence_top_k=evidence_top_k,
            support_top_k=support_top_k,
            object_top_k=object_top_k,
            cognitive_top_k=cognitive_top_k,
        )
        generated = self.services.answer_generator.generate(
            query,
            run_result,
            mode=answer_mode,
            include_creative=include_creative,
        )
        generated["route"] = getattr(run_result.get("evidence"), "query_route", {})
        return generated

    def writeback_memory(
        self,
        node: MemoryNode,
        source_kind: str | None = None,
        source_path: str | None = None,
        replace_node_id: str | None = None,
    ) -> dict[str, Any]:
        return self.services.writeback.writeback_memory(
            node=node,
            source_kind=source_kind,
            source_path=source_path,
            replace_node_id=replace_node_id,
        )
