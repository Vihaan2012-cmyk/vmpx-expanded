<#
.SYNOPSIS
  Compile the generated SP6 mutation benchmark corpus and (optionally) protect
  it with VMProtect in the requested mode.

.DESCRIPTION
  Step 1 (always): compile gen_corpus.c -> gen_corpus.dll with the first
  available compiler (clang, then gcc, then MSVC cl). The exported functions
  f000.. are the clean native code.

  Step 2 (needs VMProtect): protect the DLL in -Mode {virt|mutation|ultra}.
  VMProtect is the user's tool and is not run in CI; if VMProtect_Con.exe is not
  found the script compiles only and prints the manual protect command.

  The protected binary is the obfuscated side the tournament / WP5 native-lift
  consume. gen_oracle.json (from gen-oracle.py) is the clean-side ground truth.

.EXAMPLE
  ./build-gen.ps1 -Mode mutation
#>
param(
  [ValidateSet('virt', 'mutation', 'ultra')]
  [string]$Mode = 'mutation',
  [string]$Source = (Join-Path $PSScriptRoot 'gen_corpus.c'),
  [string]$OutDir = $PSScriptRoot,
  [string]$VMProtect = $env:VMPROTECT_CON  # path to VMProtect_Con.exe
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path $Source)) {
  throw "Source $Source not found. Run: python gen-corpus.py <N>"
}
$dll = Join-Path $OutDir 'gen_corpus.dll'

function Find-Compiler {
  foreach ($c in 'clang', 'gcc') {
    $p = (Get-Command $c -ErrorAction SilentlyContinue)
    if ($p) { return @{ kind = $c; exe = $p.Source } }
  }
  $cl = (Get-Command cl -ErrorAction SilentlyContinue)
  if ($cl) { return @{ kind = 'cl'; exe = $cl.Source } }
  throw 'No C compiler found (looked for clang, gcc, cl).'
}

$cc = Find-Compiler
Write-Host "[build-gen] compiling $Source with $($cc.kind)"
switch ($cc.kind) {
  'cl' { & $cc.exe /nologo /O2 /LD $Source /Fe:$dll | Out-Host }
  default { & $cc.exe -shared -O2 -o $dll $Source | Out-Host }
}
if ($LASTEXITCODE -ne 0 -or -not (Test-Path $dll)) { throw 'compile failed' }
Write-Host "[build-gen] built $dll"

if (-not $VMProtect) {
  $cand = (Get-Command 'VMProtect_Con.exe' -ErrorAction SilentlyContinue)
  if ($cand) { $VMProtect = $cand.Source }
}

$protected = Join-Path $OutDir ("gen_corpus.$Mode.dll")
if ($VMProtect -and (Test-Path $VMProtect)) {
  # A .vmp project selects the compilation type (Mutation/Virtualization/Ultra)
  # for the marked functions. build-gen expects gen_corpus.$Mode.vmp beside the
  # source; see benchmark.md for how it is produced.
  $proj = Join-Path $OutDir ("gen_corpus.$Mode.vmp")
  Write-Host "[build-gen] protecting ($Mode) via $VMProtect"
  if (Test-Path $proj) {
    & $VMProtect $dll $protected -pf $proj | Out-Host
  } else {
    & $VMProtect $dll $protected | Out-Host
  }
  if ($LASTEXITCODE -ne 0) { throw 'VMProtect failed' }
  Write-Host "[build-gen] protected -> $protected"
} else {
  Write-Host "[build-gen] VMProtect not found; compiled only."
  Write-Host "[build-gen] manual protect ($Mode mode):"
  Write-Host "  VMProtect_Con.exe `"$dll`" `"$protected`" -pf gen_corpus.$Mode.vmp"
}
