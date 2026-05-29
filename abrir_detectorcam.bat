@echo off
cd /d "%~dp0"
where node >nul 2>nul
if errorlevel 1 (
  echo No se encontro Node.js. Instala Node.js o abre una terminal donde node funcione.
  pause
  exit /b 1
)
start "" "http://127.0.0.1:8787"
node server.js
pause
