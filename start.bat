@echo off
cd /d "%~dp0"

echo ============================================
echo   Meta-Agent with More Agents
echo ============================================

REM Try multiple methods to find Python:
REM Method 1: LOCALAPPDATA (handles Chinese usernames safely)
for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python3*") do (
    if exist "%%D\python.exe" set PYTHON_EXE=%%D\python.exe
)

REM Method 2: py launcher (most reliable)
if not defined PYTHON_EXE (
    where py >nul 2>nul
    if not errorlevel 1 set PYTHON_EXE=py
)

REM Method 3: python from PATH
if not defined PYTHON_EXE (
    where python >nul 2>nul
    if not errorlevel 1 set PYTHON_EXE=python
)

if not defined PYTHON_EXE (
    echo [X] Python not found
    echo Install from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Starting server...
echo Open browser to: http://localhost:8000
echo ============================================
echo.

"%PYTHON_EXE%" server.py
pause
