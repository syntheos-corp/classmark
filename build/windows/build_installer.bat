@echo off
REM ============================================================================
REM Classmark Windows Installer Build Script
REM
REM This script builds the Windows installer using Inno Setup
REM
REM Prerequisites:
REM   1. Build the application first: build_windows.bat
REM   2. Install Inno Setup: https://jrsoftware.org/isdl.php
REM   3. (Optional) Download models: python download_models.py
REM
REM Usage:
REM   build_installer.bat
REM
REM Author: Classmark Development Team
REM Date: 2025-11-10
REM ============================================================================

echo.
echo ========================================================================
echo Classmark Windows Installer Build Script
echo ========================================================================
echo.

REM Check if application is built
if not exist "dist\Classmark\Classmark.exe" (
    echo [ERROR] Application not built
    echo.
    echo Please build the application first:
    echo   build_windows.bat
    echo.
    pause
    exit /b 1
)

echo [OK] Application found: dist\Classmark\Classmark.exe
echo.

REM Check if models are downloaded
if exist "models\models_config.json" (
    echo [OK] Models found - will be included in installer
    echo      Installer size: ~1-2 GB with models
) else (
    echo [WARNING] Models not found
    echo           Installer will be created without models
    echo           Users will need to download models after installation
    echo           Installer size: ~500 MB without models
    echo.
    echo To include models in installer:
    echo   python download_models.py
    echo.
)

REM Find Inno Setup compiler
set ISCC=""

REM Check common installation locations
if exist "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" (
    set ISCC="C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
)
if exist "C:\Program Files\Inno Setup 6\ISCC.exe" (
    set ISCC="C:\Program Files\Inno Setup 6\ISCC.exe"
)

REM Check if ISCC was found
if %ISCC%=="" (
    echo [ERROR] Inno Setup not found
    echo.
    echo Please install Inno Setup from:
    echo   https://jrsoftware.org/isdl.php
    echo.
    echo After installation, run this script again.
    pause
    exit /b 1
)

echo [OK] Inno Setup found: %ISCC%
echo.

REM Create output directory
if not exist "Output" mkdir Output

REM Build installer
echo Building installer...
echo This may take several minutes...
echo.

%ISCC% classmark_installer.iss

if errorlevel 1 (
    echo.
    echo [ERROR] Installer build failed
    echo Please check the error messages above
    pause
    exit /b 1
)

echo.
echo ========================================================================
echo Installer Build Complete!
echo ========================================================================
echo.
echo Installer: Output\ClassmarkSetup.exe
echo.
echo To test the installer:
echo   1. Run Output\ClassmarkSetup.exe
echo   2. Follow the installation wizard
echo   3. Launch Classmark from Start Menu or Desktop
echo.

if not exist "models\models_config.json" (
    echo IMPORTANT: Models were NOT included in this installer
    echo Users will need to download models after installation:
    echo   1. Open Command Prompt as Administrator
    echo   2. Navigate to installation directory
    echo   3. Run: python download_models.py
    echo.
)

pause
