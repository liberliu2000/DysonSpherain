from __future__ import annotations

import json
import plistlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1] / "base"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dysonspherain.adapters.supervisor import install_all, install_supervisor, render_launchd_plist, render_systemd_service, status_all, uninstall_all


class SupervisorTests(unittest.TestCase):
    def test_launchd_plist_contains_daemon_command_and_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            payload = plistlib.loads(
                render_launchd_plist(
                    "memory-daemon",
                    project,
                    python="/usr/bin/python3",
                    port=37888,
                    project_name="DysonSpherain",
                )
            )
            self.assertEqual(payload["Label"], "ai.dysonspherain.dysonspherain.memory-daemon")
            self.assertIn("dysonspherain.adapters.daemon", payload["ProgramArguments"])
            self.assertIn("--port", payload["ProgramArguments"])
            self.assertIn("37888", payload["ProgramArguments"])
            self.assertIn("PYTHONPATH", payload["EnvironmentVariables"])
            self.assertTrue(payload["KeepAlive"])

    def test_systemd_service_contains_scheduler_daemon_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            text = render_systemd_service(
                "memory-scheduler",
                project,
                python="/usr/bin/python3",
                interval_seconds=2.5,
            )
            self.assertIn("[Service]", text)
            self.assertIn("sphere_cli.cli", text)
            self.assertIn("memory scheduler --daemon", text)
            self.assertIn("--interval-seconds 2.5", text)
            self.assertIn("Restart=on-failure", text)

    def test_install_supervisor_is_idempotent_without_activation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "home"
            project = Path(tmp) / "project"
            project.mkdir()
            with patch.object(Path, "home", return_value=root):
                first = install_supervisor(project, service="memory-daemon", platform_name="launchd", python=sys.executable)
                second = install_supervisor(project, service="memory-daemon", platform_name="launchd", python=sys.executable)
                self.assertEqual(first.status, "ok")
                self.assertTrue(first.changed)
                self.assertFalse(second.changed)
                self.assertTrue(Path(first.path).exists())
                all_status = status_all(platform_name="launchd")
                self.assertEqual(len(all_status), 2)
                self.assertTrue(any(row["installed"] for row in all_status))

    def test_install_and_uninstall_all_systemd_configs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "home"
            project = Path(tmp) / "project"
            project.mkdir()
            with patch.object(Path, "home", return_value=root):
                installed = install_all(project, platform_name="systemd", python=sys.executable)
                self.assertEqual(len(installed), 2)
                self.assertTrue(all(row["status"] == "ok" for row in installed))
                self.assertTrue(all(Path(row["path"]).exists() for row in installed))
                removed = uninstall_all(platform_name="systemd")
                self.assertTrue(all(row["changed"] for row in removed))
                self.assertFalse(any(Path(row["path"]).exists() for row in removed))

    def test_cli_installs_supervisor_configs_without_activation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            project = Path(tmp) / "project"
            project.mkdir()
            script = (
                "from pathlib import Path\n"
                "from unittest.mock import patch\n"
                "import sys\n"
                f"with patch.object(Path, 'home', return_value=Path({str(home)!r})):\n"
                "    from sphere_cli.cli import app\n"
                "    import typer\n"
                f"    sys.argv = ['dysonspherain', 'adapters', 'install-supervisor', '--project', {str(project)!r}, '--platform', 'launchd', '--python', sys.executable]\n"
                "    app()\n"
            )
            proc = subprocess.run([sys.executable, "-c", script], cwd=ROOT.parent, env={"PYTHONPATH": str(ROOT)}, text=True, capture_output=True, check=True)
            payload = json.loads(proc.stdout)
            self.assertEqual(len(payload), 2)
            self.assertTrue(all(row["status"] == "ok" for row in payload))


if __name__ == "__main__":
    unittest.main()
