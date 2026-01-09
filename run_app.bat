@echo off
echo ==================================================
echo   HOUSE PRICE PREDICTION - LITE EDITION
echo ==================================================
echo.
echo [1/3] Checking libraries...
pip install -r requirements.txt > nul
echo.
echo [2/3] Training/Updating Model...
python train_lite.py
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ TRAINING FAILED! Please check the errors above.
    pause
    exit
)
echo.
echo [3/3] Launching App...
echo.
echo NOTE: Browser will open automatically.
echo       Press CTRL+C here to stop the server.
echo.
python -m streamlit run app.py
pause