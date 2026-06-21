#!/bin/bash

# RL Agent UI - Quick Start Script for Linux/Mac

echo ""
echo "====================================================="
echo "  RL Agent - Radiation Therapy UI Launcher"
echo "====================================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8+ and try again"
    exit 1
fi

echo "[*] Python found:"
python3 --version

# Check if Flask is installed
python3 -c "import flask" 2>/dev/null
if [ $? -ne 0 ]; then
    echo ""
    echo "[!] Dependencies not found. Installing..."
    pip3 install -r requirements_ui.txt
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install dependencies"
        exit 1
    fi
fi

echo ""
echo "[*] All dependencies OK"
echo ""
echo "[*] Starting Flask Backend Server..."
echo "    - API: http://localhost:5000"
echo "    - Open ui.html in your browser"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python3 app.py --config configs/default.yaml --ckpt runs/best.pt --port 5000
