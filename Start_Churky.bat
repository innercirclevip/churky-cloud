@echo off
cd C:\churky
call .venv\Scripts\activate
echo Starting Churky Server...
start "Churky Server" cmd /k "python churky_server.py"
echo Starting Ngrok Tunnel...
start "Ngrok Tunnel" cmd /k "ngrok http --url=hazelnut-anew-grain.ngrok-free.dev 8000"