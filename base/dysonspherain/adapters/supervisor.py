from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import platform
import plistlib
import shutil
import subprocess
import sys
from typing import Any


SERVICE_COMMANDS = {
    "memory-daemon": ["-m", "dysonspherain.adapters.daemon"],
    "memory-scheduler": ["-m", "sphere_cli.cli", "memory", "scheduler", "--daemon"],
}


@dataclass(frozen=True)
class SupervisorResult:
    status: str
    platform: str
    service: str
    path: str
    changed: bool = False
    activated: bool = False
    commands: list[list[str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    backup_path: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def supported_platform() -> str:
    system = platform.system().lower()
    if system == "darwin":
        return "launchd"
    if system == "linux":
        return "systemd"
    return "unsupported"


def _python_path(python: str | None = None) -> str:
    return str(Path(python or sys.executable).resolve())


def _base_path(project: Path) -> Path:
    return project / "base"


def _logs_dir(project: Path) -> Path:
    path = project / "data" / "supervisor" / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _service_env(project: Path) -> dict[str, str]:
    base = _base_path(project)
    existing = os.environ.get("PYTHONPATH", "")
    pythonpath = str(base) if not existing else f"{base}{os.pathsep}{existing}"
    return {
        "PYTHONPATH": pythonpath,
        "DYSON_PROJECT_ROOT": str(project),
        "DYSON_HOME": str(project / ".dyson"),
    }


def service_command(service: str, project: Path, *, python: str | None = None, host: str = "127.0.0.1", port: int = 37777, project_name: str = "DysonSpherain", interval_seconds: float = 5.0) -> list[str]:
    if service not in SERVICE_COMMANDS:
        raise ValueError(f"unknown_supervisor_service:{service}")
    command = [_python_path(python), *SERVICE_COMMANDS[service]]
    if service == "memory-daemon":
        command.extend(["--base-dir", str(project), "--project", project_name, "--host", host, "--port", str(port)])
    else:
        command.extend(["--project", project_name, "--interval-seconds", str(interval_seconds)])
    return command


def launchd_label(service: str, project_name: str = "DysonSpherain") -> str:
    normalized_project = "".join(ch.lower() if ch.isalnum() else "-" for ch in project_name).strip("-") or "dysonspherain"
    return f"ai.dysonspherain.{normalized_project}.{service}"


def systemd_unit_name(service: str, project_name: str = "DysonSpherain") -> str:
    normalized_project = "".join(ch.lower() if ch.isalnum() else "-" for ch in project_name).strip("-") or "dysonspherain"
    return f"dysonspherain-{normalized_project}-{service}.service"


def render_launchd_plist(service: str, project: Path, *, python: str | None = None, host: str = "127.0.0.1", port: int = 37777, project_name: str = "DysonSpherain", interval_seconds: float = 5.0) -> bytes:
    logs = _logs_dir(project)
    payload = {
        "Label": launchd_label(service, project_name),
        "ProgramArguments": service_command(service, project, python=python, host=host, port=port, project_name=project_name, interval_seconds=interval_seconds),
        "WorkingDirectory": str(project),
        "EnvironmentVariables": _service_env(project),
        "RunAtLoad": True,
        "KeepAlive": True,
        "StandardOutPath": str(logs / f"{service}.out.log"),
        "StandardErrorPath": str(logs / f"{service}.err.log"),
    }
    return plistlib.dumps(payload, sort_keys=True)


def render_systemd_service(service: str, project: Path, *, python: str | None = None, host: str = "127.0.0.1", port: int = 37777, project_name: str = "DysonSpherain", interval_seconds: float = 5.0) -> str:
    command = " ".join(_systemd_quote(part) for part in service_command(service, project, python=python, host=host, port=port, project_name=project_name, interval_seconds=interval_seconds))
    env = _service_env(project)
    logs = _logs_dir(project)
    lines = [
        "[Unit]",
        f"Description=DysonSpherain {service}",
        "After=network.target",
        "",
        "[Service]",
        "Type=simple",
        f"WorkingDirectory={project}",
        *(f"Environment={key}={_systemd_env_quote(value)}" for key, value in env.items()),
        f"ExecStart={command}",
        "Restart=on-failure",
        "RestartSec=5",
        f"StandardOutput=append:{logs / f'{service}.out.log'}",
        f"StandardError=append:{logs / f'{service}.err.log'}",
        "",
        "[Install]",
        "WantedBy=default.target",
        "",
    ]
    return "\n".join(lines)


def _systemd_quote(value: str) -> str:
    if all(ch.isalnum() or ch in "/._:=+-" for ch in value):
        return value
    return "'" + value.replace("'", "'\\''") + "'"


def _systemd_env_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def supervisor_path(service: str, *, platform_name: str | None = None, project_name: str = "DysonSpherain") -> Path:
    kind = platform_name or supported_platform()
    if kind == "launchd":
        return Path.home() / "Library" / "LaunchAgents" / f"{launchd_label(service, project_name)}.plist"
    if kind == "systemd":
        return Path.home() / ".config" / "systemd" / "user" / systemd_unit_name(service, project_name)
    raise ValueError(f"unsupported_supervisor_platform:{kind}")


def install_supervisor(
    project: Path,
    *,
    service: str,
    platform_name: str | None = None,
    python: str | None = None,
    host: str = "127.0.0.1",
    port: int = 37777,
    project_name: str = "DysonSpherain",
    interval_seconds: float = 5.0,
    activate: bool = False,
) -> SupervisorResult:
    project = project.resolve()
    kind = platform_name or supported_platform()
    if kind not in {"launchd", "systemd"}:
        return SupervisorResult("unsupported", kind, service, "", warnings=["Only macOS launchd and Linux systemd user services are supported."])
    path = supervisor_path(service, platform_name=kind, project_name=project_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        render_launchd_plist(service, project, python=python, host=host, port=port, project_name=project_name, interval_seconds=interval_seconds)
        if kind == "launchd"
        else render_systemd_service(service, project, python=python, host=host, port=port, project_name=project_name, interval_seconds=interval_seconds).encode("utf-8")
    )
    old = path.read_bytes() if path.exists() else b""
    changed = old != content
    backup_path = ""
    if changed:
        if path.exists():
            backup = path.with_suffix(path.suffix + ".bak")
            backup.write_bytes(old)
            backup_path = str(backup)
        path.write_bytes(content)
    commands: list[list[str]] = []
    warnings: list[str] = []
    activated = False
    if activate:
        commands, warnings, activated = _activate(kind, path, service=service, project_name=project_name)
    return SupervisorResult("ok", kind, service, str(path), changed=changed, activated=activated, commands=commands, warnings=warnings, backup_path=backup_path)


def uninstall_supervisor(*, service: str, platform_name: str | None = None, project_name: str = "DysonSpherain", deactivate: bool = False) -> SupervisorResult:
    kind = platform_name or supported_platform()
    if kind not in {"launchd", "systemd"}:
        return SupervisorResult("unsupported", kind, service, "", warnings=["Only macOS launchd and Linux systemd user services are supported."])
    path = supervisor_path(service, platform_name=kind, project_name=project_name)
    commands: list[list[str]] = []
    warnings: list[str] = []
    activated = False
    if deactivate and path.exists():
        commands, warnings, activated = _deactivate(kind, path, service=service, project_name=project_name)
    changed = path.exists()
    path.unlink(missing_ok=True)
    return SupervisorResult("ok", kind, service, str(path), changed=changed, activated=activated, commands=commands, warnings=warnings)


def supervisor_status(*, service: str, platform_name: str | None = None, project_name: str = "DysonSpherain") -> dict[str, Any]:
    kind = platform_name or supported_platform()
    if kind not in {"launchd", "systemd"}:
        return {"status": "unsupported", "platform": kind, "service": service}
    path = supervisor_path(service, platform_name=kind, project_name=project_name)
    payload: dict[str, Any] = {"status": "ok", "platform": kind, "service": service, "path": str(path), "installed": path.exists()}
    if kind == "launchd":
        label = launchd_label(service, project_name)
        payload["label"] = label
        payload["loaded"] = _command_ok(["launchctl", "print", f"gui/{os.getuid()}/{label}"])
    else:
        unit = systemd_unit_name(service, project_name)
        payload["unit"] = unit
        payload["active"] = _command_ok(["systemctl", "--user", "is-active", "--quiet", unit])
        payload["enabled"] = _command_ok(["systemctl", "--user", "is-enabled", "--quiet", unit])
    return payload


def install_all(project: Path, **kwargs: Any) -> list[dict[str, Any]]:
    return [install_supervisor(project, service=service, **kwargs).to_dict() for service in SERVICE_COMMANDS]


def uninstall_all(**kwargs: Any) -> list[dict[str, Any]]:
    return [uninstall_supervisor(service=service, **kwargs).to_dict() for service in SERVICE_COMMANDS]


def status_all(**kwargs: Any) -> list[dict[str, Any]]:
    return [supervisor_status(service=service, **kwargs) for service in SERVICE_COMMANDS]


def _activate(kind: str, path: Path, *, service: str, project_name: str) -> tuple[list[list[str]], list[str], bool]:
    if kind == "launchd":
        commands = [["launchctl", "bootstrap", f"gui/{os.getuid()}", str(path)], ["launchctl", "enable", f"gui/{os.getuid()}/{launchd_label(service, project_name)}"]]
    else:
        unit = systemd_unit_name(service, project_name)
        commands = [["systemctl", "--user", "daemon-reload"], ["systemctl", "--user", "enable", "--now", unit]]
    return _run_commands(commands)


def _deactivate(kind: str, path: Path, *, service: str, project_name: str) -> tuple[list[list[str]], list[str], bool]:
    if kind == "launchd":
        commands = [["launchctl", "bootout", f"gui/{os.getuid()}", str(path)]]
    else:
        unit = systemd_unit_name(service, project_name)
        commands = [["systemctl", "--user", "disable", "--now", unit], ["systemctl", "--user", "daemon-reload"]]
    return _run_commands(commands)


def _run_commands(commands: list[list[str]]) -> tuple[list[list[str]], list[str], bool]:
    warnings: list[str] = []
    if any(shutil.which(command[0]) is None for command in commands):
        missing = sorted({command[0] for command in commands if shutil.which(command[0]) is None})
        return commands, [f"missing supervisor command(s): {', '.join(missing)}"], False
    for command in commands:
        proc = subprocess.run(command, text=True, capture_output=True)
        if proc.returncode != 0:
            warnings.append(f"{' '.join(command)} failed: {(proc.stderr or proc.stdout).strip()}")
            return commands, warnings, False
    return commands, warnings, True


def _command_ok(command: list[str]) -> bool:
    if shutil.which(command[0]) is None:
        return False
    return subprocess.run(command, text=True, capture_output=True).returncode == 0
