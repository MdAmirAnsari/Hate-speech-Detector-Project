@echo off
echo Starting Hate Speech Detection Web Interface...
echo.
echo Make sure you have activated the virtual environment first:
echo ..\hate-speech-env\Scripts\Activate.ps1
echo.
echo Starting Streamlit...
streamlit run streamlit_app.py --server.port 8501 --server.address localhost
pause
