# Buggy calculator

`calculator.py` deliberately subtracts instead of adding. The existing unittest
suite checks positive, negative, zero, and fractional inputs. No target-project
dependencies are required beyond Python.

Use a separate copy as its own clean Git repository. The example directory inside
the Cleanroom checkout is not itself a Git root. From the Cleanroom repository
root in PowerShell:

```powershell
$demoRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("cleanroom-calculator-" + [guid]::NewGuid().ToString("N"))
Copy-Item -LiteralPath .\examples\buggy_calculator -Destination $demoRoot -Recurse
git -C $demoRoot init
git -C $demoRoot add .
git -C $demoRoot -c user.name="Cleanroom Demo" -c user.email="demo@example.invalid" -c commit.gpgsign=false commit -m "Broken calculator example"

$python = (Resolve-Path .\.venv\Scripts\python.exe).Path
& $python -m harness run --repo $demoRoot --task "Fix add so it correctly adds two numbers" --output .\runs\calculator-001 --editable calculator.py -- $python -m unittest -v
```

Install `.[dev,claude]` and set `ANTHROPIC_API_KEY` and `CLEANROOM_MODEL` first.
The `run` command makes live model API requests. The output directory must be new;
choose another name for subsequent runs.
The original example remains broken because edits are saved as a reviewable patch
from a disposable copy. Inspect `report.json`, `transcript.json`, and `changes.patch`.
