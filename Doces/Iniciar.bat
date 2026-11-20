@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" main.py
) else (
    python main.py
)
if errorlevel 1 (
    echo.
    echo Nao foi possivel iniciar. Consulte LEIAME.txt para instalar as dependencias.
    pause
)
