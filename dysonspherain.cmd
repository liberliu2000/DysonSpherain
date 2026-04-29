@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "PYTHON_BIN=%DYSONSPHERAIN_PYTHON%"
if not defined PYTHON_BIN set "PYTHON_BIN=python"

set "PYTHONPATH=%SCRIPT_DIR%overlay;%SCRIPT_DIR%base;%PYTHONPATH%"
"%PYTHON_BIN%" -m sphere_cli %*
