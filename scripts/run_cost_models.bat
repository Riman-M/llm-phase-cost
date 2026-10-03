@echo off
REM ===================================================================
REM  run_cost_models.bat
REM
REM  Runs cost_model.py over all four Azure traces under three runtime
REM  configurations measured in a single Kaggle session:
REM
REM     fp32   PyTorch eager, float32        (joint_torch.csv)
REM     f16    llama.cpp, 16-bit weights     (joint_llama.csv)
REM     q4_k_m llama.cpp, 4-bit weights      (joint_llama.csv)
REM
REM  Each configuration writes to its own output directory, because
REM  cost_model.py always names its output cost_model.csv and a shared
REM  directory would leave only the last run standing.
REM
REM  Everything printed is also appended to cost_models_log.txt.
REM
REM  Place beside cost_model.py, with the traces in .\data\ and the two
REM  profile CSVs downloaded from the notebook output.
REM ===================================================================

setlocal enabledelayedexpansion
set LOG=cost_models_log.txt

echo. > "%LOG%"
echo ================================================== >> "%LOG%"
echo  Cost model sweep  %DATE% %TIME% >> "%LOG%"
echo ================================================== >> "%LOG%"

REM --- preflight -----------------------------------------------------
REM Checking first is worth the lines: the 2024 traces take minutes to
REM stream, and discovering a missing file on the third run wastes them.

set MISSING=0
for %%F in (cost_model.py joint_torch.csv joint_llama.csv) do (
    if not exist "%%F" (
        echo [MISSING] %%F
        set MISSING=1
    )
)
for %%F in (2023_conv 2023_code 2024_conv 2024_code) do (
    if not exist "data\%%F.csv" (
        echo [MISSING] data\%%F.csv
        set MISSING=1
    )
)
if "!MISSING!"=="1" (
    echo.
    echo One or more inputs are missing. See the list above.
    echo   profile CSVs  - download from the Kaggle notebook output
    echo   data\*.csv    - the four Azure traces, named as above
    goto :end
)

python --version >nul 2>&1
if errorlevel 1 (
    echo Python was not found on PATH.
    goto :end
)

echo All inputs present. Starting sweep.
echo.

REM --- the three runs ------------------------------------------------

call :run fp32   joint_torch.csv  "Qwen/Qwen2.5-1.5B"     results_fp32
call :run f16    joint_llama.csv  Qwen2.5-1.5B-f16        results_f16
call :run q4     joint_llama.csv  Qwen2.5-1.5B-q4_k_m     results_q4

echo.
echo ===================================================================
echo  Sweep finished.
echo.
echo  Per-run output:  results_fp32\cost_model.csv
echo                   results_f16\cost_model.csv
echo                   results_q4\cost_model.csv
echo  Combined log:    %LOG%
echo ===================================================================
goto :end

REM --- subroutine: one configuration ---------------------------------
REM %1 label  %2 profile csv  %3 model id  %4 output dir

:run
echo.
echo ------------------------------------------------------------------
echo  %~1   model %~3
echo ------------------------------------------------------------------
echo. >> "%LOG%"
echo ===== %~1  (%~3) ===== >> "%LOG%"

python cost_model.py --profile %~2 --traces data/*.csv --model %~3 --threads 4 --outdir %~4 2>&1 | python -c "import sys;[ (sys.stdout.write(l), open(r'%LOG%','a',encoding='utf-8').write(l)) for l in sys.stdin ]"

if errorlevel 1 (
    echo   [FAILED] %~1 - see %LOG%
    echo   [FAILED] %~1 >> "%LOG%"
) else (
    echo   [ok] %~1
)
exit /b 0

:end
echo.
pause
endlocal
