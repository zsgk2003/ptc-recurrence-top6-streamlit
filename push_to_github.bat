@echo off
REM Push this app to GitHub (run after creating the empty repo on GitHub).
setlocal
cd /d "%~dp0"

set REPO=https://github.com/zsgk2003/ptc-recurrence-top6-streamlit.git

echo ============================================================
echo  PTC Top-6 Streamlit App -> GitHub
echo  Target: %REPO%
echo ============================================================
echo.
echo Before running:
echo   1. Open https://github.com/new
echo   2. Repository name: ptc-recurrence-top6-streamlit
echo   3. Public, NO template, do NOT add README
echo   4. Create repository
echo.
pause

if not exist ".git" (
    echo [ERROR] No git repo. Run from 02_PTC_top6_streamlit_app after git init.
    pause
    exit /b 1
)

git remote remove origin 2>nul
git remote add origin %REPO%

echo Pushing branch main...
git push -u origin main
if errorlevel 1 (
    echo.
    echo [FAILED] Push did not complete. Common fixes:
    echo   - Create the empty repo on GitHub first
    echo   - Sign in: git credential-manager or use a Personal Access Token
    echo   - Check network / VPN to github.com
    pause
    exit /b 1
)

echo.
echo [OK] Pushed to %REPO%
echo.
echo Next: deploy on Streamlit Community Cloud
echo   https://share.streamlit.io/
echo   Repo: zsgk2003/ptc-recurrence-top6-streamlit
echo   Main file: app.py
echo.
pause
