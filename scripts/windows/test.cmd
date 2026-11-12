@echo off
rem test.cmd of the Windows package (scripts/windows/package.ps1): the tiered KV cache exactness test (scripts\tiered_cache_test.py)
rem with the package's Python, on the weights start.cmd downloads. Arguments go to the test (--scenarios a,c). docs/deploy-windows.md.
rem Stop the server first (tray: Stop server, or Ctrl+C in start.cmd's window): the test loads its own engine.
setlocal
cd /d "%~dp0"
python\python.exe app\scripts\tiered_cache_test.py %*
set "RC=%errorlevel%"
echo.
echo exit code %RC%
pause
exit /b %RC%
