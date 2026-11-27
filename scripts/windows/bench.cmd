@echo off
rem bench.cmd of the Windows package (scripts/windows/package.ps1): the tiered KV cache timings (scripts\tiered_cache_bench.py)
rem with the package's Python. The SSD files go to a bench folder of the kvcache folder (the bench deletes the files of its entries\ and chunks\ folders
rem there). Arguments are added: --sizes 32768 for one size, or --compare "SPILL_PINNED_GIB=0" "SPILL_PINNED_GIB=8" for
rem variants (docs/deploy-windows.md). Stop the server first.
rem The SSD tier is off by default in the server (SPILL_SSD_GIB=0), but the scenarios that use it (and the bench) need it: --ssd-gib 10 turns it on
rem for this run, and --ssd-gib 0 makes them fail instead of passing without the SSD. Your own --ssd-gib later in the arguments wins.
setlocal
set "PYTHONUTF8=1"
cd /d "%~dp0"
python\python.exe app\scripts\tiered_cache_bench.py --spill-dir "%LOCALAPPDATA%\qwen3-27b-5090\kvcache\bench" --ssd-gib 10 %*
set "RC=%errorlevel%"
echo.
echo exit code %RC%
pause
exit /b %RC%
