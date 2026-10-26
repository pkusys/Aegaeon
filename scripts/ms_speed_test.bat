@echo off
rem Windows launcher for ms_speed_test.py
setlocal
where python >nul 2>nul
if %errorlevel%==0 (
  python "%~dp0ms_speed_test.py" %*
  exit /b %errorlevel%
)
where py >nul 2>nul
if %errorlevel%==0 (
  py -3 "%~dp0ms_speed_test.py" %*
  exit /b %errorlevel%
)
echo Python not found. Install Python 3 from https://www.python.org/downloads/
pause
exit /b 1
endlocal
