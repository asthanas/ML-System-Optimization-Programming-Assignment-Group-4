@echo off
REM Compressed-DDP Setup Script (Windows)
REM For Linux/macOS, use: bash setup.sh or python setup.py

echo ======================================================================
echo Compressed-DDP Setup (Windows)
echo ======================================================================
echo.

REM Check Python version
echo [1/5] Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found in PATH
    echo Please install Python 3.9+ from python.org
    exit /b 1
)
echo OK Python found

REM Create virtual environment
echo.
echo [2/5] Creating virtual environment...
if not exist venv (
    python -m venv venv
    echo OK Virtual environment created
) else (
    echo OK Virtual environment exists
)

REM Activate and upgrade pip
echo.
echo [3/5] Upgrading pip...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip -q
echo OK Pip upgraded

REM Install requirements
echo.
echo [4/5] Installing requirements...
if not exist requirements.txt (
    echo Error: requirements.txt not found
    exit /b 1
)
pip install -r requirements.txt -q
echo OK Requirements installed

REM Install package
echo.
echo [5/5] Installing package in editable mode...
pip install -e . -q
echo OK Package installed

REM Check GPU
echo.
echo Checking GPU availability...
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU - CPU mode')"

echo.
echo ======================================================================
echo Setup complete!
echo ======================================================================
echo.
echo Next steps:
echo   1. Activate environment: venv\Scripts\activate
echo   2. Run validation: python experiments\quick_validation.py
echo   3. Run tests: pytest tests\
echo   4. Train model: python train.py --help
echo.
echo ======================================================================
pause
