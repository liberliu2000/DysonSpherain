#!/usr/bin/env node
"use strict";

const fs = require("fs");
const path = require("path");
const { spawnSync } = require("child_process");

const repoRoot = path.resolve(__dirname, "..");
const basePath = path.join(repoRoot, "base");
const bootstrapVenv = path.join(repoRoot, ".dyson-quickstart-venv");

function usage() {
  console.log(`DysonSpherain Memory quick-start wrapper

Usage:
  npx dysonspherain-memory install [--project <path>]
  npm install -g dysonspherain-memory && dyson-memory install --project .
  npx dysonspherain-memory doctor [--project <path>]
  npx dysonspherain-memory plugin install [--project <path>]
  npx dysonspherain-memory plugin print
  npx dysonspherain-memory daemon [--project <path>] [--port <port>]
  npx dysonspherain-memory supervisor install [--project <path>] [--activate]
  npx dysonspherain-memory supervisor status [--project <path>]
  npx dysonspherain-memory supervisor uninstall [--deactivate]
  npx dysonspherain-memory bootstrap
  npx dysonspherain-memory mcp-smoke

Options:
  --python <exe>     Python executable to use (default: DYSON_PYTHON, python3, python)
  --project <path>   Project root (default: current directory)
  --port <port>      Daemon port (default: 37777)
  --no-bootstrap     Do not create the package-local Python environment
`);
}

function parseArgs(argv) {
  const args = { _: [] };
  for (let i = 0; i < argv.length; i += 1) {
    const item = argv[i];
    if (item.startsWith("--")) {
      const key = item.slice(2);
      const next = argv[i + 1];
      if (next && !next.startsWith("--")) {
        args[key] = next;
        i += 1;
      } else {
        args[key] = true;
      }
    } else {
      args._.push(item);
    }
  }
  return args;
}

function executableWorks(command) {
  const result = spawnSync(command, ["--version"], { encoding: "utf8" });
  return result.status === 0;
}

function pythonEnv(extraEnv = {}) {
  const env = { ...process.env, ...extraEnv };
  env.PYTHONPATH = env.PYTHONPATH ? `${basePath}${path.delimiter}${env.PYTHONPATH}` : basePath;
  return env;
}

function resolvePython(explicit) {
  const candidates = [explicit, process.env.DYSON_PYTHON, "python3", "python"].filter(Boolean);
  for (const candidate of candidates) {
    if (executableWorks(candidate)) return candidate;
  }
  throw new Error("No usable Python executable found. Pass --python <exe> or set DYSON_PYTHON.");
}

function venvPython() {
  if (process.platform === "win32") {
    return path.join(bootstrapVenv, "Scripts", "python.exe");
  }
  return path.join(bootstrapVenv, "bin", "python");
}

function pythonCanImport(python) {
  const probe = spawnSync(
    python,
    ["-c", "import sphere_cli.cli, dysonspherain.adapters.mcp_server"],
    { cwd: repoRoot, env: pythonEnv(), encoding: "utf8" }
  );
  return probe.status === 0;
}

function runChecked(command, args, options = {}) {
  const result = spawnSync(command, args, { stdio: "inherit", ...options });
  if (result.status !== 0) {
    process.exit(result.status || 1);
  }
}

function bootstrapPython(seedPython, force = false) {
  const targetPython = venvPython();
  if (!force && fs.existsSync(targetPython) && pythonCanImport(targetPython)) {
    return targetPython;
  }
  fs.mkdirSync(bootstrapVenv, { recursive: true });
  console.error(`[dysonspherain-memory] Preparing Python runtime at ${bootstrapVenv}`);
  runChecked(seedPython, ["-m", "venv", bootstrapVenv], { cwd: repoRoot });
  runChecked(targetPython, ["-m", "pip", "install", "--upgrade", "pip"], { cwd: repoRoot });
  runChecked(targetPython, ["-m", "pip", "install", "."], { cwd: repoRoot });
  return targetPython;
}

function resolveRuntimePython(args, command) {
  const seed = resolvePython(args.python);
  if (args["no-bootstrap"] || process.env.DYSON_NO_BOOTSTRAP === "1") {
    return seed;
  }
  if (pythonCanImport(seed)) {
    return seed;
  }
  if (command === "help") {
    return seed;
  }
  return bootstrapPython(seed, Boolean(args.force));
}

function runPython(python, moduleArgs, options = {}) {
  const result = spawnSync(python, ["-m", "sphere_cli.cli", ...moduleArgs], {
    cwd: options.cwd || repoRoot,
    env: pythonEnv(),
    stdio: "inherit",
  });
  if (result.status !== 0) {
    process.exit(result.status || 1);
  }
}

function runModule(python, moduleName, moduleArgs, options = {}) {
  const result = spawnSync(python, ["-m", moduleName, ...moduleArgs], {
    cwd: options.cwd || repoRoot,
    env: pythonEnv(),
    stdio: "inherit",
  });
  if (result.status !== 0) {
    process.exit(result.status || 1);
  }
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const command = args._[0] || "help";
  if (command === "help" || args.help || args.h) {
    usage();
    return;
  }
  if (!fs.existsSync(basePath)) {
    throw new Error(`Cannot find DysonSpherain Python package at ${basePath}`);
  }
  if (command === "bootstrap") {
    bootstrapPython(resolvePython(args.python), true);
    return;
  }
  const python = resolveRuntimePython(args, command);
  const project = path.resolve(args.project || process.cwd());

  if (command === "install") {
    runPython(python, ["adapters", "install-codex-mcp", "--project", project], { cwd: repoRoot });
    runPython(python, ["adapters", "install-claude-hooks", "--project", project], { cwd: repoRoot });
    runPython(python, ["adapters", "install-plugin-manifests", "--project", project], { cwd: repoRoot });
    runPython(python, ["adapters", "doctor", "--project", project], { cwd: repoRoot });
    return;
  }

  if (command === "doctor") {
    runPython(python, ["adapters", "doctor", "--project", project], { cwd: repoRoot });
    return;
  }

  if (command === "plugin") {
    const sub = args._[1] || "install";
    if (sub === "install") {
      runPython(python, ["adapters", "install-plugin-manifests", "--project", project], { cwd: repoRoot });
      return;
    }
    if (sub === "print") {
      const manifestPath = path.join(repoRoot, ".codex-plugin", "plugin.json");
      process.stdout.write(fs.readFileSync(manifestPath, "utf8"));
      return;
    }
    if (sub === "path") {
      console.log(path.join(repoRoot, ".codex-plugin", "plugin.json"));
      return;
    }
    console.error(`Unknown plugin command: ${sub}`);
    usage();
    process.exit(2);
  }

  if (command === "daemon") {
    runPython(
      python,
      ["adapters", "daemon", "--project-root", project, "--project", args.projectName || "DysonSpherain", "--port", String(args.port || 37777)],
      { cwd: repoRoot }
    );
    return;
  }

  if (command === "supervisor") {
    const sub = args._[1] || "status";
    if (sub === "install") {
      const cmd = ["adapters", "install-supervisor", "--project", project, "--python", python, "--project-name", args.projectName || "DysonSpherain", "--port", String(args.port || 37777)];
      if (args.activate) cmd.push("--activate");
      if (args.platform) cmd.push("--platform", args.platform);
      runPython(python, cmd, { cwd: repoRoot });
      return;
    }
    if (sub === "status") {
      const cmd = ["adapters", "supervisor-status", "--project-name", args.projectName || "DysonSpherain"];
      if (args.platform) cmd.push("--platform", args.platform);
      runPython(python, cmd, { cwd: repoRoot });
      return;
    }
    if (sub === "uninstall") {
      const cmd = ["adapters", "uninstall-supervisor", "--project-name", args.projectName || "DysonSpherain"];
      if (args.deactivate) cmd.push("--deactivate");
      if (args.platform) cmd.push("--platform", args.platform);
      runPython(python, cmd, { cwd: repoRoot });
      return;
    }
    console.error(`Unknown supervisor command: ${sub}`);
    usage();
    process.exit(2);
  }

  if (command === "mcp-smoke") {
    runModule(python, "dysonspherain.adapters.mcp_server", ["--smoke"], { cwd: repoRoot });
    return;
  }

  console.error(`Unknown command: ${command}`);
  usage();
  process.exit(2);
}

try {
  main();
} catch (error) {
  console.error(error.message || String(error));
  process.exit(1);
}
