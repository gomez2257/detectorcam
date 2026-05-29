@echo off
setlocal
cd /d "%~dp0"

where node >nul 2>nul
if errorlevel 1 (
  echo No se encontro Node.js en este equipo.
  echo Instala Node.js desde https://nodejs.org/ y vuelve a abrir este archivo.
  pause
  exit /b 1
)

start "" "http://127.0.0.1:8787"
node server.js
pause
