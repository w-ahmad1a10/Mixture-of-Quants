@echo off
setlocal
cd /d "%~dp0"
start "Mixture of Quants Watcher" /MIN python git_watcher.py
echo Watcher started in the background.
