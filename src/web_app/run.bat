@echo off
echo ============================================================
echo CaloScan - Food Nutrition Analyzer Web App
echo ============================================================
echo.
echo Activating conda environment...
call conda activate torch-gpu

echo.
echo Starting Flask server...
echo.
echo The app will be available at: http://localhost:5000
echo Press CTRL+C to stop the server
echo.
echo ============================================================
echo.

python app.py

pause
