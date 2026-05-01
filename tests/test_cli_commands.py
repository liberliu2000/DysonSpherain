from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "base"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sphere_cli.cli import app


class CliCommandRegistrationTests(unittest.TestCase):
    def test_root_ask_command_has_single_citation_first_implementation(self) -> None:
        ask_commands = [
            command
            for command in app.registered_commands
            if command.name == "ask"
        ]

        self.assertEqual(len(ask_commands), 1)
        self.assertEqual(ask_commands[0].callback.__name__, "ask")

    def test_root_recall_command_uses_memory_os_interface(self) -> None:
        recall_commands = [
            command
            for command in app.registered_commands
            if command.name == "recall"
        ]
        legacy_commands = [
            command
            for command in app.registered_commands
            if command.name == "legacy-recall"
        ]

        self.assertEqual(len(recall_commands), 1)
        self.assertEqual(recall_commands[0].callback.__name__, "dyson_recall_command")
        self.assertEqual(len(legacy_commands), 1)

    def test_runs_typer_is_registered(self) -> None:
        typer_names = {group.name for group in app.registered_groups}

        self.assertIn("runs", typer_names)

    def test_agent_typer_and_memory_state_typer_are_registered(self) -> None:
        root_typer_names = {group.name for group in app.registered_groups}
        memory_group = next(group for group in app.registered_groups if group.name == "memory")
        memory_typer_names = {group.name for group in memory_group.typer_instance.registered_groups}

        self.assertIn("agent", root_typer_names)
        self.assertIn("state", memory_typer_names)
        self.assertIn("conflicts", memory_typer_names)
        self.assertIn("lifecycle", memory_typer_names)

    def test_memory_context_command_and_agent_preflight_are_registered(self) -> None:
        memory_group = next(group for group in app.registered_groups if group.name == "memory")
        agent_group = next(group for group in app.registered_groups if group.name == "agent")
        memory_commands = {command.name for command in memory_group.typer_instance.registered_commands}
        agent_commands = {command.name for command in agent_group.typer_instance.registered_commands}

        self.assertIn("remember", memory_commands)
        self.assertIn("search", memory_commands)
        self.assertIn("inspect", memory_commands)
        self.assertIn("update", memory_commands)
        self.assertIn("archive", memory_commands)
        self.assertIn("context", memory_commands)
        self.assertIn("debug-context", memory_commands)
        self.assertIn("why-selected", memory_commands)
        self.assertIn("preflight", agent_commands)
        self.assertIn("status", agent_commands)

    def test_agent_ledger_typer_is_registered(self) -> None:
        agent_group = next(group for group in app.registered_groups if group.name == "agent")
        agent_typer_names = {group.name for group in agent_group.typer_instance.registered_groups}

        self.assertIn("ledger", agent_typer_names)

    def test_code_typer_is_registered(self) -> None:
        root_typer_names = {group.name for group in app.registered_groups}
        code_group = next(group for group in app.registered_groups if group.name == "code")
        code_commands = {command.name for command in code_group.typer_instance.registered_commands}

        self.assertIn("code", root_typer_names)
        self.assertIn("index", code_commands)
        self.assertIn("search-symbol", code_commands)
        self.assertIn("relevant-files", code_commands)
