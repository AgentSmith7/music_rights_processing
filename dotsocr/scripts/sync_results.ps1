# Auto-sync DotsOCR results from RunPod to local
# Run this in a separate terminal: .\sync_results.ps1

$remoteHost = "root@103.207.149.99"
$remotePort = "11116"
$sshKey = "~/.ssh/id_ed25519"
$remotePath = "/workspace/music_rights/data/output_dotsocr_smart/*.json"
$localPath = "C:\Users\risha\cursorprojects\WeaveAssignment\document-agent\music_rights\data\output_dotsocr_smart\"
$interval = 60  # seconds

Write-Host "=== DotsOCR Results Auto-Sync ===" -ForegroundColor Cyan
Write-Host "Remote: $remoteHost`:$remotePath"
Write-Host "Local: $localPath"
Write-Host "Interval: ${interval}s"
Write-Host ""

while ($true) {
    $timestamp = Get-Date -Format "HH:mm:ss"
    
    # Get count before sync
    $beforeCount = (Get-ChildItem $localPath -Filter "*.json" -ErrorAction SilentlyContinue | Measure-Object).Count
    
    # Sync files
    scp -P $remotePort -i $sshKey "${remoteHost}:${remotePath}" $localPath 2>$null
    
    # Get count after sync
    $afterCount = (Get-ChildItem $localPath -Filter "*.json" -ErrorAction SilentlyContinue | Measure-Object).Count
    $newFiles = $afterCount - $beforeCount
    
    if ($newFiles -gt 0) {
        Write-Host "[$timestamp] Downloaded $newFiles new file(s) - Total: $afterCount/23" -ForegroundColor Green
    } else {
        Write-Host "[$timestamp] No new files - Total: $afterCount/23" -ForegroundColor Gray
    }
    
    # Check if done
    if ($afterCount -ge 23) {
        Write-Host ""
        Write-Host "=== All 23 PDFs downloaded! ===" -ForegroundColor Green
        break
    }
    
    Start-Sleep -Seconds $interval
}
