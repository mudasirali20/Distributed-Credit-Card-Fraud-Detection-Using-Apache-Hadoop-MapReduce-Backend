@echo off
title Hadoop Fraud Detection Web Platform
echo ===================================================
echo Starting Distributed Fraud Detection Web Platform
echo ===================================================
echo.

echo Starting FastAPI Backend (Port 8000)...
start "Fraud Detection Backend (Port 8000)" cmd /k "cd /d "%~dp0backend" && python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload"

echo Starting React + Vite Frontend (Port 5173)...
start "Fraud Detection Frontend (Port 5173)" cmd /k "cd /d "%~dp0frontend" && npm.cmd run dev"

echo Waiting for servers to initialize...
timeout /t 3 >nul

echo Opening website in your default browser...
start http://localhost:5173

echo.
echo All services launched!
echo Press any key to exit this launcher window (servers will remain running).
pause >nul
