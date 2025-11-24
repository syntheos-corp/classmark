@echo off
REM ============================================================================
REM Classmark Windows Build Script
REM
REM This script builds the Windows executable using PyInstaller
REM
REM Usage:
REM   build_windows.bat [clean]
REM
REM Options:
REM   clean - Clean build directories before building
REM
REM Author: Classmark Development Team
REM Date: 2025-11-10
REM ============================================================================

echo.
echo ========================================================================
echo Classmark Windows Build Script
echo ========================================================================
echo.

REM Check if PyInstaller is installed
python -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo [ERROR] PyInstaller is not installed
    echo.
    echo Installing PyInstaller...
    pip install pyinstaller
    if errorlevel 1 (
        echo [ERROR] Failed to install PyInstaller
        exit /b 1
    )
)

REM Clean build directories if requested
if "%1"=="clean" (
    echo Cleaning build directories...
    if exist build rmdir /s /q build
    if exist dist rmdir /s /q dist
    echo [OK] Build directories cleaned
    echo.
)

REM Check if requirements are installed
echo Checking dependencies...
python -c "import torch, transformers, ultralytics, tkinter, ttkbootstrap" 2>nul
if errorlevel 1 (
    echo [WARNING] Some dependencies may be missing
    echo Please run: pip install -r requirements.txt
    echo.
    echo Note: ttkbootstrap is required for modern UI themes
    pause
)

REM Build the executable
echo.
echo Building Windows executable...
echo This may take several minutes (5-15 minutes depending on your system)
echo.
pyinstaller classmark.spec

if errorlevel 1 (
    echo.
    echo [ERROR] Build failed
    echo Please check the error messages above
    pause
    exit /b 1
)

echo.
echo ========================================================================
echo Build Complete!
echo ========================================================================
echo.
echo Output directory: dist\Classmark\
echo Executable: dist\Classmark\Classmark.exe
echo.
echo To test the application:
echo   cd dist\Classmark
echo   Classmark.exe
echo.
echo To create an installer:
echo   Run build_installer.bat
echo.

pause
