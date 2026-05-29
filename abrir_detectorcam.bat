@echo off
cd /d "%~dp0"
echo Iniciando DetectorCam...
echo.
where node >nul 2>nul
if errorlevel 1 (
  echo No se encontro Node.js.
  echo Instala Node.js desde https://nodejs.org/ y vuelve a intentar.
  pause
  exit /b 1
)
start "" "http://127.0.0.1:8787"
node server.js
pause
