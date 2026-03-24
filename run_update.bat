@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

set PYTHON=C:\Users\jt060\AppData\Local\Python\pythoncore-3.14-64\python.exe
set STREAMLIT=C:\Users\jt060\AppData\Local\Python\pythoncore-3.14-64\Scripts\streamlit.exe
set GIT=git

if not exist logs mkdir logs

for /f "tokens=*" %%d in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd"') do set TODAY=%%d
set LOGFILE=logs\update_%TODAY:.=%.log

echo [%date% %time%] START >> "%LOGFILE%"

rem ----------------------------------------------------------
rem  Step 1: data update
rem ----------------------------------------------------------
echo [%time%] [1/3] data update ... >> "%LOGFILE%"

"%PYTHON%" scripts\update_data.py --mode all >> "%LOGFILE%" 2>&1

if %ERRORLEVEL% NEQ 0 (
    echo [%date% %time%] ERROR: data update failed >> "%LOGFILE%"
    exit /b 1
)

echo [%time%] [1/3] data update OK >> "%LOGFILE%"

rem ----------------------------------------------------------
rem  Step 2: git add / commit / push
rem ----------------------------------------------------------
echo [%time%] [2/3] git push ... >> "%LOGFILE%"

%GIT% add data/ >> "%LOGFILE%" 2>&1

%GIT% diff --cached --quiet
if !ERRORLEVEL! NEQ 0 (
    %GIT% commit -m "data: auto update %TODAY%" >> "%LOGFILE%" 2>&1
    %GIT% push origin main >> "%LOGFILE%" 2>&1
    set PUSH_ERR=!ERRORLEVEL!
    if !PUSH_ERR! NEQ 0 (
        echo [%time%] [2/3] WARNING: git push failed >> "%LOGFILE%"
    ) else (
        echo [%time%] [2/3] git push OK >> "%LOGFILE%"
    )
) else (
    echo [%time%] [2/3] no data changes, skip commit >> "%LOGFILE%"
)

rem ----------------------------------------------------------
rem  Step 3: restart local Streamlit
rem ----------------------------------------------------------
echo [%time%] [3/3] restarting Streamlit ... >> "%LOGFILE%"

taskkill /F /IM streamlit.exe >> "%LOGFILE%" 2>&1
timeout /t 3 /nobreak > nul

start "" /b "%STREAMLIT%" run app.py --server.headless true --server.port 8501

echo [%time%] [3/3] Streamlit started (port 8501) >> "%LOGFILE%"
echo [%date% %time%] DONE >> "%LOGFILE%"

endlocal
exit /b 0
