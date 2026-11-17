@echo off
rem Starts the interactive demo in your browser (http://localhost:8501).
cd /d "%~dp0.."
python -m streamlit run app\demo.py
