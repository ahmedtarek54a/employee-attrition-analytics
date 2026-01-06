@echo off
echo ==================================================
echo   Available Employee Attrition AI System
echo ==================================================
echo.
echo [1/2] Installing Dependencies (this may take a minute)...
pip install -r requirements.txt
echo.
echo [2/2] Launching Application...
streamlit run streamlit_app.py
pause
