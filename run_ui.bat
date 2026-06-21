@echo off
REM RL Agent UI - Quick Start Script for Windows

echo.
echo =====================================================
echo   RL Agent - Radiation Therapy UI Launcher
echo =====================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and add it to PATH
    pause
    exit /b 1
)

echo [*] Python found:
python --version

REM Check if Flask is installed
pip show flask >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [!] Flask not found. Installing dependencies...
    pip install -r requirements_ui.txt
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
)

echo.
echo [*] All dependencies OK
echo.
echo [*] Starting Flask Backend Server...
echo     - API: http://localhost:5000
echo     - Open ui.html in your browser
echo.
echo Press Ctrl+C to stop the server
echo.

python app.py --config configs/default.yaml --ckpt runs/best.pt --port 5000

pause
