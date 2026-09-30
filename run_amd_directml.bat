@echo off
cd /d "%~dp0"
where py >nul 2>nul && (py -3 start.py --desktop --directml) || (python start.py --desktop --directml)
pause
