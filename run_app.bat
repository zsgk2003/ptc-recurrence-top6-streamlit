@echo off
REM Launch the app with the existing local conda environment `radiomics`.
setlocal
set PY=D:\anaconda3\envs\radiomics\python.exe

cd /d "%~dp0"

if not exist "%PY%" (
    echo [ERROR] Python not found: %PY%
    echo Edit the PY variable in this script to point at your environment.
    pause
    exit /b 1
)

if not exist "artifacts\model_LightGBM_top6.pkl" (
    echo First run: training and exporting the model...
    "%PY%" train_model.py
)

echo Starting Streamlit at http://localhost:8501
"%PY%" -m streamlit run app.py

pause
