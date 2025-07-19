@echo off

echo 🚀 Preparing for Render deployment...

REM Add all files
git add .

REM Create commit message with timestamp
for /f "tokens=1-4 delims=/ " %%i in ('date /t') do set mydate=%%i-%%j-%%k-%%l
for /f "tokens=1-2 delims=: " %%i in ('time /t') do set mytime=%%i:%%j
set commit_message=Deploy: %mydate% %mytime%

REM Commit
git commit -m "%commit_message%"

echo 📦 Files committed with message: %commit_message%

REM Push to main branch
git push origin main

echo ✅ Pushed to GitHub - Render will automatically deploy!
echo 🔗 Check your Render dashboard for deployment progress
echo 📱 Your API will be live at: https://your-app-name.onrender.com

pause
