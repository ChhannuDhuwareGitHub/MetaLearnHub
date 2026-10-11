@echo off
echo =======================================
echo Starting MetaLearnHub...
echo =======================================

echo Starting Backend API...
start cmd /k "cd c:\projects\project9(mega)\metalearnhub && python backend/api/main.py"

echo Starting Frontend Dev Server...
start cmd /k "cd c:\projects\project9(mega)\metalearnhub\frontend && npm run dev"

echo MetaLearnHub is starting! The frontend will be available at http://localhost:5173
echo The backend is running at http://localhost:8000
