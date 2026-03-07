@echo off
REM start.bat — One-click launcher for MEV Bot Dashboard on Windows
REM
REM Double-click this file OR run from Command Prompt / PowerShell.
REM Options are passed straight through to run.py:
REM   start.bat --port 8080
REM   start.bat --no-browser

:: Move to the folder containing this script
cd /d "%~dp0"

:: Try python, then py launcher
where python >nul 2>&1 && (
    python run.py %*
    goto :end
)
where py >nul 2>&1 && (
    py run.py %*
    goto :end
)

echo.
echo ERROR: Python 3.10+ is required.
echo Download from https://www.python.org/downloads/
echo.
pause
exit /b 1

:end
pause
