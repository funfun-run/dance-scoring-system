@echo off
cd /d "%~dp0"
.venv\Scripts\python.exe -c "from dance_scoring.gui.app import MainApp; MainApp().run()"
pause
