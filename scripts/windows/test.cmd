@echo off
rem test.cmd of the Windows package (scripts/windows/package.ps1): the tiered KV cache exactness test (scripts\tiered_cache_test.py)
rem with the package's Python, on the weights start.cmd downloads. Arguments go to the test (--scenarios a,c). docs/deploy-windows.md.
rem The image and video API test is the other way round (the server must be running): python\python.exe app\scripts\vision_api_test.py --base-url http://127.0.0.1:8080
rem Stop the server first (tray: Stop server, or Ctrl+C in start.cmd's window): the test loads its own engine.
rem The SSD tier is off by default in the server (SPILL_SSD_GIB=0), but the scenarios that use it (and the bench) need it: --ssd-gib 10 turns it on
rem for this run, and --ssd-gib 0 makes them fail instead of passing without the SSD. Your own --ssd-gib later in the arguments wins.
setlocal
set "PYTHONUTF8=1"
cd /d "%~dp0"
python\python.exe app\scripts\tiered_cache_test.py --ssd-gib 10 %*
set "RC=%errorlevel%"
echo.
echo exit code %RC%
pause
exit /b %RC%
