[CmdletBinding()]
param(
    [ValidateRange(1, 1024)]
    [double]$MaxUsedGb = 55
)

$pcbrOperatingSystem = Get-CimInstance Win32_OperatingSystem
$pcbrTotalGb = [math]::Round($pcbrOperatingSystem.TotalVisibleMemorySize / 1MB, 2)
$pcbrFreeGb = [math]::Round($pcbrOperatingSystem.FreePhysicalMemory / 1MB, 2)
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
