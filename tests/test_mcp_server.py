from __future__ import annotations

import anyio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1] / "base"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dysonspherain.adapters.mcp_server import TOOLS, TOOL_SCHEMAS, call_tool, handle_jsonrpc_request, mcp_sdk_available, smoke_payload
from dysonspherain.memory_os.observation_store import token_economy_summary


class McpServerTests(unittest.TestCase):
    def test_smoke_lists_required_tools(self) -> None:
        payload = smoke_payload()
        self.assertEqual(payload["status"], "ok")
        self.assertIn("serverInfo", payload)
        self.assertIn("protocolVersion", payload)
        self.assertIn("transport_implementation", payload)
        self.assertEqual(payload["mcp_sdk_available"], mcp_sdk_available())
        self.assertEqual(set(payload["tools"]), set(TOOLS))
        self.assertIn("dyson_memory_intent", payload["tools"])
        self.assertIn("dyson_search_memory", payload["tools"])

    def test_token_economy_tool_returns_decision(self) -> None:
        result = call_tool(
            "dyson_token_economy_eval",
            {
                "query": "fix benchmark regression",
                "candidate_context": "Benchmark regression evidence and file references",
                "token_budget": 1600,
                "task_type": "debug",
            },
        )
        self.assertEqual(result["status"], "ok")
        self.assertIn(result["decision"], {"inject", "skip", "inject_summary_only", "return_file_refs_only"})

    def test_token_economy_eval_can_write_ledger_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"DYSON_ALLOWED_PATHS": tmp}):
                result = call_tool(
                    "dyson_token_economy_eval",
                    {
                        "cwd": tmp,
                        "project": "DysonSpherain",
                        "session_id": "mcp-ledger",
                        "query": "repair token economy ledger writeback",
                        "candidate_context": "ledger writeback evidence and file references",
                        "baseline_context_tokens": 1200,
                        "token_budget": 600,
                        "write_ledger": True,
                    },
                )
                summary = token_economy_summary(Path(tmp), project="DysonSpherain")
        self.assertEqual(result["status"], "ok")
        self.assertIn("ledger_write", result)
        self.assertEqual(result["ledger_write"]["status"], "ok")
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(len(summary["events"]), 1)
        self.assertEqual(summary["events"][0]["adapter"], "codex_mcp")

    def test_tools_list_exposes_input_schemas(self) -> None:
        self.assertEqual(set(TOOL_SCHEMAS), set(TOOLS))
        self.assertIn("prompt", TOOL_SCHEMAS["dyson_memory_intent"]["properties"])
        self.assertIn("query", TOOL_SCHEMAS["dyson_recall"]["properties"])
        self.assertIn("memory_ids", TOOL_SCHEMAS["dyson_context_pack"]["properties"])
        self.assertIn("candidates", TOOL_SCHEMAS["dyson_context_pack"]["properties"])
        self.assertIn("ranked_items", TOOL_SCHEMAS["dyson_context_pack"]["properties"])
        self.assertIn("memory_objects", TOOL_SCHEMAS["dyson_context_pack"]["properties"])
        self.assertIn("candidate_context", TOOL_SCHEMAS["dyson_token_economy_eval"]["properties"])
        self.assertIn("cwd", TOOL_SCHEMAS["dyson_token_economy_eval"]["properties"])
        self.assertIn("write_ledger", TOOL_SCHEMAS["dyson_token_economy_eval"]["properties"])
        self.assertIn("query", TOOL_SCHEMAS["dyson_search_memory"]["properties"])
        self.assertIn("observation_id", TOOL_SCHEMAS["dyson_timeline"]["properties"])
        self.assertIn("observation_ids", TOOL_SCHEMAS["dyson_get_observations"]["properties"])
        self.assertIn("lookback_hours", TOOL_SCHEMAS["dyson_resume_context"]["properties"])
        self.assertIn("text", TOOL_SCHEMAS["dyson_product_write"]["properties"])
        self.assertIn("show_audit", TOOL_SCHEMAS["dyson_product_retrieve"]["properties"])
        self.assertIn("validity_state", TOOL_SCHEMAS["dyson_product_update_validity"]["properties"])
        self.assertIn("task", TOOL_SCHEMAS["dyson_runtime_before_task"]["properties"])
        self.assertIn("artifact", TOOL_SCHEMAS["dyson_benchmark_record"]["properties"])
        self.assertIn("current", TOOL_SCHEMAS["dyson_benchmark_compare"]["properties"])

    def test_memory_intent_tool_routes_continuation_to_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"DYSON_ALLOWED_PATHS": tmp}):
                result = call_tool("dyson_memory_intent", {"cwd": tmp, "prompt": "继续"})
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["should_call_memory"])
        self.assertEqual(result["reason"], "cross_session_continuation")
        self.assertIn("dyson_resume_context", result["recommended_tools"])

    def test_context_pack_accepts_supplied_candidates(self) -> None:
        result = call_tool(
            "dyson_context_pack",
            {
                "candidates": [{"id": "c1", "text": "Supplied ranked memory object.", "path": "base/x.py", "score": 0.8}],
                "sections": ["core_evidence", "relevant_files"],
                "format": "markdown",
                "token_budget": 800,
            },
        )
        self.assertEqual(result["status"], "ok")
        self.assertIn("Supplied ranked memory object.", result["rendered_context"])
        self.assertIn("base/x.py", result["rendered_context"])

    def test_write_memory_tool_redacts_and_dedupes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = {
                "cwd": tmp,
                "session_id": "s1",
                "task_goal": "store run",
                "summary": "Fixed issue with api_key=sk-abcdef1234567890",
                "files_changed": [],
                "commands_run": [],
                "tests_run": [],
                "benchmark_results": [],
                "failures": [],
                "next_actions": [],
                "source": "codex",
            }
            with patch.dict(os.environ, {"DYSON_ALLOWED_PATHS": tmp}):
                first = call_tool("dyson_write_memory", args)
                second = call_tool("dyson_write_memory", args)
            self.assertEqual(first["status"], "ok")
            self.assertEqual(second["status"], "duplicate")
            self.assertTrue(first["sanitizer"]["has_redaction"])

    def test_progressive_observation_tools_search_timeline_and_get(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"DYSON_ALLOWED_PATHS": tmp}):
                write = call_tool(
                    "dyson_write_memory",
                    {
                        "cwd": tmp,
                        "session_id": "s-progressive",
                        "task_goal": "progressive recall",
                        "summary": "CloneMem candidate_recall regression note",
                        "files_changed": [],
                        "commands_run": [],
                        "tests_run": [],
                        "benchmark_results": [],
                        "failures": [],
                        "next_actions": [],
                        "source": "manual",
                    },
                )
                self.assertEqual(write["status"], "ok")
                search = call_tool("dyson_search_memory", {"cwd": tmp, "query": "candidate_recall", "limit": 5})
                self.assertEqual(search["count"], 1)
                obs_id = search["observations"][0]["observation_id"]
                details = call_tool("dyson_get_observations", {"cwd": tmp, "observation_ids": [obs_id]})
                self.assertIn("candidate_recall", details["observations"][0]["content"])
                events = call_tool("dyson_timeline", {"cwd": tmp, "observation_id": obs_id})
                self.assertGreaterEqual(len(events["events"]), 1)
                resume = call_tool("dyson_resume_context", {"cwd": tmp, "session_id": "s-progressive"})
                self.assertEqual(resume["status"], "ok")
                self.assertIn("CloneMem candidate_recall", resume["rendered_context"])

    def test_product_mcp_tools_cover_capsules_runtime_benchmark_and_health(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"DYSON_ALLOWED_PATHS": tmp}):
                first = call_tool(
                    "dyson_product_write",
                    {
                        "cwd": tmp,
                        "project": "P",
                        "text": "MCP product capsule benchmark decision",
                        "evidence_type": "decision",
                        "tags": ["mcp"],
                    },
                )
                second = call_tool(
                    "dyson_product_write",
                    {
                        "cwd": tmp,
                        "project": "P",
                        "text": "MCP product capsule replacement decision",
                        "evidence_type": "decision",
                        "tags": ["mcp"],
                    },
                )
                self.assertEqual(first["status"], "ok")
                search = call_tool("dyson_product_search", {"cwd": tmp, "project": "P", "query": "benchmark decision", "limit": 5})
                self.assertGreaterEqual(search["count"], 1)
                retrieved = call_tool("dyson_product_retrieve", {"cwd": tmp, "project": "P", "query": "benchmark decision", "show_audit": True, "context_pack": True})
                self.assertIn("retrieval_trace", retrieved)
                wake = call_tool("dyson_product_wake", {"cwd": tmp, "project": "P", "task": "benchmark decision", "format": "text"})
                self.assertIn("rendered", wake)
                inspected = call_tool("dyson_product_inspect", {"cwd": tmp, "project": "P", "capsule_id": first["capsule_id"]})
                self.assertEqual(inspected["capsule"]["id"], first["capsule_id"])
                validity = call_tool(
                    "dyson_product_update_validity",
                    {
                        "cwd": tmp,
                        "project": "P",
                        "capsule_id": first["capsule_id"],
                        "validity_state": "superseded",
                        "by_capsule_id": second["capsule_id"],
                        "reason": "newer decision",
                    },
                )
                self.assertEqual(validity["validity_state"], "superseded")
                runtime = call_tool("dyson_runtime_before_task", {"cwd": tmp, "project": "P", "task": "run product MCP smoke"})
                self.assertEqual(runtime["status"], "ok")
                compact = call_tool("dyson_runtime_pre_compact", {"cwd": tmp, "project": "P", "session_id": "s1"})
                self.assertEqual(compact["event"], "pre_compact")

                current = Path(tmp) / "current.json"
                baseline = Path(tmp) / "baseline.json"
                current.write_text(json.dumps({"benchmark": "unit", "metrics": {"recall": 0.8, "candidate_recall@100": 0.9}}), encoding="utf-8")
                baseline.write_text(json.dumps({"benchmark": "unit", "metrics": {"recall": 0.7, "candidate_recall@100": 0.95}}), encoding="utf-8")
                recorded = call_tool("dyson_benchmark_record", {"cwd": tmp, "project": "P", "artifact": str(current), "benchmark": "unit"})
                self.assertEqual(recorded["benchmark"], "unit")
                compared = call_tool("dyson_benchmark_compare", {"cwd": tmp, "project": "P", "current": str(current), "baseline": str(baseline)})
                self.assertIn("candidate_recall_deltas", compared)
                health = call_tool("dyson_health_doctor", {"cwd": tmp, "project": "P"})
                self.assertIn("checks", health)

    def test_mcp_rejects_cwd_outside_allowed_roots(self) -> None:
        response = handle_jsonrpc_request(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {"name": "dyson_context_pack", "arguments": {"cwd": "/tmp/dyson_mcp_outside_root", "query": "q"}},
            }
        )
        self.assertIsNotNone(response)
        assert response is not None
        self.assertEqual(response["error"]["code"], -32000)
        self.assertIn("path_outside_allowed_roots", response["error"]["data"])

    def test_jsonrpc_initialize_exposes_mcp_metadata(self) -> None:
        response = handle_jsonrpc_request({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        self.assertIsNotNone(response)
        assert response is not None
        result = response["result"]
        self.assertIn("protocolVersion", result)
        self.assertEqual(result["serverInfo"]["name"], "dyson-memory")
        self.assertIn("tools", result["capabilities"])
        self.assertIn("implementation", result)
        self.assertIn("transport_implementation", result["implementation"])

    def test_jsonrpc_errors_are_structured(self) -> None:
        invalid = handle_jsonrpc_request({"jsonrpc": "2.0", "id": 2, "method": "missing/method"})
        self.assertIsNotNone(invalid)
        assert invalid is not None
        self.assertEqual(invalid["error"]["code"], -32601)
        unknown_tool = handle_jsonrpc_request({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "nope", "arguments": {}}})
        self.assertIsNotNone(unknown_tool)
        assert unknown_tool is not None
        self.assertEqual(unknown_tool["error"]["code"], -32602)

    def test_mcp_sdk_stdio_server_lists_tools_and_calls_tool(self) -> None:
        if not mcp_sdk_available():
            self.skipTest("mcp SDK is not installed")

        async def run_client() -> None:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            params = StdioServerParameters(
                command=sys.executable,
                args=["-m", "dysonspherain.adapters.mcp_server"],
                cwd=str(ROOT.parent),
                env={"PYTHONPATH": str(ROOT), "DYSON_ALLOWED_PATHS": str(ROOT.parent)},
            )
            async with stdio_client(params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    names = {tool.name for tool in tools.tools}
                    self.assertEqual(names, set(TOOLS))
                    context_schema = next(tool.inputSchema for tool in tools.tools if tool.name == "dyson_context_pack")
                    self.assertIn("candidates", context_schema["properties"])
                    search_schema = next(tool.inputSchema for tool in tools.tools if tool.name == "dyson_search_memory")
                    self.assertIn("query", search_schema["properties"])
                    resume_schema = next(tool.inputSchema for tool in tools.tools if tool.name == "dyson_resume_context")
                    self.assertIn("token_budget", resume_schema["properties"])
                    intent_schema = next(tool.inputSchema for tool in tools.tools if tool.name == "dyson_memory_intent")
                    self.assertIn("prompt", intent_schema["properties"])
                    result = await session.call_tool(
                        "dyson_token_economy_eval",
                        {
                            "query": "repair benchmark regression",
                            "candidate_context": "candidate recall and ndcg regression evidence",
                            "token_budget": 1600,
                        },
                    )
                    self.assertFalse(result.isError)
                    self.assertTrue(result.content)

        anyio.run(run_client)


if __name__ == "__main__":
    unittest.main()
