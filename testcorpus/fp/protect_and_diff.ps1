# Builds the FP test lib (GNU toolchain), protects every export with VMProtect
# (virt mode, same options as bigcorpus/protect_all.py) and runs the ground-truth
# differential. Static file operations only: nothing here executes the built DLL.
#
#   .\protect_and_diff.ps1 -Vmpx E:\scratch\target\release\vmpx.exe -Work E:\scratch\fp
param(
    [Parameter(Mandatory = $true)][string]$Vmpx,
    [Parameter(Mandatory = $true)][string]$Work,
    [string]$Con = "E:\VMPX\testcorpus\vmprotect\install\VMProtect_Con.exe",
    # fp_pow_iat only exists to exercise the call path; the VM-side call handler
    # does not recover on every build, so it is excluded by default.
    [string[]]$Exports = @("fp_arith", "fp_sqrt", "fp_cvt", "fp_f32", "fp_powf", "fp_trunc", "fp_neg")
)
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force $Work | Out-Null
Copy-Item "$PSScriptRoot\Cargo.toml", "$PSScriptRoot\src" -Destination $Work -Recurse -Force
New-Item -ItemType Directory -Force "$Work\.cargo" | Out-Null
Set-Content "$Work\.cargo\config.toml" "[net]`noffline = true`n"
Push-Location $Work
$env:CARGO_INCREMENTAL = "0"
cargo +stable-x86_64-pc-windows-gnu build --release
Pop-Location
$orig = "$Work\target\release\fplib.dll"
$prot = "$Work\fplib.virt.dll"
$procs = ($Exports | ForEach-Object { "            <Procedure MapAddress=`"$_`" Options=`"0`" CompilationType=`"0`" />" }) -join "`n"
$xml = @"
<?xml version="1.0" encoding="UTF-8" ?>
<Document Version="2">
    <Protection InputFileName="$orig" Options="262144" VMCodeSectionName=".???" VMComplexity="20" OutputFileName="$prot">
        <Procedures>
$procs
        </Procedures>
    </Protection>
</Document>
"@
Set-Content "$Work\fplib.virt.vmp" $xml -Encoding UTF8
& $Con "$Work\fplib.virt.vmp"
# Features needed for the z3 backends: --features smt-z3,smt-z3-bundled
& $Vmpx diff-original $prot $orig
# Negative control: every function against the NEXT export's original.
& $Vmpx diff-original $prot $orig --control
