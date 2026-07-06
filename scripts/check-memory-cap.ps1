[CmdletBinding()]
param(
    [ValidateRange(1, 1024)]
    [double]$MaxUsedGb = 55
)

try {
    $pcbrOperatingSystem = Get-CimInstance Win32_OperatingSystem -ErrorAction Stop
    $pcbrTotalBytes = [double]$pcbrOperatingSystem.TotalVisibleMemorySize * 1KB
    $pcbrFreeBytes = [double]$pcbrOperatingSystem.FreePhysicalMemory * 1KB
}
catch {
    try {
        # CIM can be denied in restricted build agents even though the process may read the
        # equivalent kernel counters. ComputerInfo keeps the guard usable without weakening it.
        Add-Type -AssemblyName Microsoft.VisualBasic -ErrorAction Stop
        $pcbrComputerInfo = [Microsoft.VisualBasic.Devices.ComputerInfo]::new()
        $pcbrTotalBytes = [double]$pcbrComputerInfo.TotalPhysicalMemory
        $pcbrFreeBytes = [double]$pcbrComputerInfo.AvailablePhysicalMemory
    }
    catch {
        Write-Error "Unable to query host memory; refusing to treat the memory cap as satisfied."
        exit 2
    }
}

if (
    -not [double]::IsFinite($pcbrTotalBytes) -or
    -not [double]::IsFinite($pcbrFreeBytes) -or
    $pcbrTotalBytes -le 0 -or
    $pcbrFreeBytes -lt 0 -or
    $pcbrFreeBytes -gt $pcbrTotalBytes
) {
    Write-Error "Host memory telemetry is invalid; refusing to treat the memory cap as satisfied."
    exit 2
}

$pcbrTotalGb = [math]::Round($pcbrTotalBytes / 1GB, 2)
$pcbrFreeGb = [math]::Round($pcbrFreeBytes / 1GB, 2)
$pcbrUsedGb = [math]::Round($pcbrTotalGb - $pcbrFreeGb, 2)
$pcbrWithinCap = $pcbrUsedGb -lt $MaxUsedGb

[pscustomobject]@{
    used_gb = $pcbrUsedGb
    free_gb = $pcbrFreeGb
    total_gb = $pcbrTotalGb
    max_used_gb = $MaxUsedGb
    within_cap = $pcbrWithinCap
} | ConvertTo-Json -Compress

if (-not $pcbrWithinCap) {
    Write-Error "System memory usage is $pcbrUsedGb GB, at or above the $MaxUsedGb GB cap."
    exit 1
}
