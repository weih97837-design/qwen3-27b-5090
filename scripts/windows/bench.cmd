@echo off
rem bench.cmd of the Windows package (scripts/windows/package.ps1): the tiered KV cache timings (scripts\tiered_cache_bench.py)
rem with the package's Python. The SSD files go to a bench folder of the kvcache folder (the bench deletes the files of its entries\ and chunks\ folders
rem there). Arguments are added: --sizes 32768 for one size, or --compare "SPILL_PINNED_GIB=0" "SPILL_PINNED_GIB=8" for
rem variants (docs/deploy-windows.md). Stop the server first.
setlocal
cd /d "%~dp0"
python\python.exe app\scripts\tiered_cache_bench.py --spill-dir "%LOCALAPPDATA%\qwen3-27b-5090\kvcache\bench" %*
set "RC=%errorlevel%"
echo.
echo exit code %RC%
pause
exit /b %RC%
