@echo off
cd /d "%~dp0"
start "" "http://127.0.0.1:8787"
"C:\Users\porta\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe" server.js
