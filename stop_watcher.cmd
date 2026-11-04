@echo off
setlocal
cd /d "%~dp0"
if not exist "watcher.lock" (
    echo Watcher is not running.
    exit /b 0
)
set /p WATCHER_PID=<watcher.lock
taskkill /PID %WATCHER_PID% /F >nul 2>&1
del "watcher.lock" >nul 2>&1
echo Watcher stopped.
