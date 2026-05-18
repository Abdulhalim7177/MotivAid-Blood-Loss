@echo off
REM Install TFLite conversion dependencies for Windows
REM Run this from your activated virtual environment

echo ============================================================
echo   Installing TFLite Conversion Dependencies
echo ============================================================
echo.

echo Uninstalling old onnx-tf if present...
pip uninstall -y onnx-tf

echo.
echo Installing required packages...
pip install onnx onnx2tf tensorflow numpy pillow

echo.
echo ============================================================
echo   Installation Complete!
echo ============================================================
echo.
echo Next step: python scripts/convert_to_tflite.py
echo.
pause
