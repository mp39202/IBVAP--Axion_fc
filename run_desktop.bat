@echo off
cd /d "%~dp0"
where py >nul 2>nul && (py -3 start.py --desktop %*) || (python start.py --desktop %*)
pause
