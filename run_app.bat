@echo off
rem Start the Streamlit UI with the project's own Python environment (.venv),
rem so a global Streamlit install (e.g. Miniconda) cannot be picked up by mistake.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo .venv not found. Create it and install requirements.txt first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run app.py --server.address localhost %*
