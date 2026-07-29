# Rebuild the project environment.
#
# The venv lives on the project drive, NOT under %TEMP%: a machine restart
# cleared %TEMP% mid-experiment once and took the interpreter with it.
#
# pandas and statsmodels are pinned because Windows Application Control blocks
# freshly published binaries until they acquire reputation — pandas 3.0.5 was
# rejected on this machine while 3.0.3 loaded fine. If an import dies with
# "Una directiva de Control de aplicaciones bloqueó este archivo", step the
# version DOWN rather than reinstalling the same one.
#
#   powershell -ExecutionPolicy Bypass -File scripts\make_env.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$proj = Split-Path -Parent $root
$venv = Join-Path $proj ".venv"
$base = "C:\Users\fmarr\AppData\Local\Programs\Python\Python311\python.exe"

if (-not (Test-Path $venv)) { & $base -m venv $venv }
$py = Join-Path $venv "Scripts\python.exe"

& $py -m pip install --upgrade pip
& $py -m pip install torch --index-url https://download.pytorch.org/whl/cu128
& $py -m pip install numpy "pandas==3.0.3" scikit-learn xgboost lightgbm `
    "statsmodels==0.14.6" python-docx EMD-signal matplotlib

& $py -c @"
import importlib
bad = []
for m in ['numpy','pandas','sklearn','xgboost','lightgbm','statsmodels',
          'docx','PyEMD','matplotlib','torch']:
    try:
        importlib.import_module(m)
    except Exception as e:
        bad.append(f'{m}: {e}')
import torch
print('cuda:', torch.cuda.is_available())
if bad:
    raise SystemExit('BROKEN: ' + ' | '.join(bad))
print('environment OK')
"@
