# ─────────────────────────────────────────────────────────────
# rgsplat — one-time install script (Windows)
# ─────────────────────────────────────────────────────────────
# Usage:  powershell -ExecutionPolicy Bypass -File install.ps1
# ─────────────────────────────────────────────────────────────

$ErrorActionPreference = "Stop"

Write-Host "=== rgsplat: installing dependencies ===" -ForegroundColor Cyan

# ── ffmpeg (winget) ──────────────────────────────────────────
if (Get-Command ffmpeg -ErrorAction SilentlyContinue) {
    Write-Host "  ffmpeg OK" -ForegroundColor Green
} else {
    Write-Host "  Installing ffmpeg via winget..." -ForegroundColor Yellow
    winget install --silent --accept-package-agreements FFmpeg
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "User") + ";$env:Path"
}

# ── COLMAP ───────────────────────────────────────────────────
if (Get-Command colmap -ErrorAction SilentlyContinue) {
    Write-Host "  COLMAP OK" -ForegroundColor Green
} else {
    Write-Host "  Installing COLMAP (CUDA binary)..." -ForegroundColor Yellow
    $url = "https://github.com/colmap/colmap/releases/download/3.9.1/colmap-3.9.1-windows-cuda.zip"
    $zip = "$env:TEMP\colmap.zip"
    $dest = "$env:LOCALAPPDATA\Programs\colmap"
    Write-Host "    Downloading COLMAP 3.9.1..." -ForegroundColor Gray
    Invoke-WebRequest -Uri $url -OutFile $zip
    Expand-Archive -Path $zip -DestinationPath $dest -Force
    $colmapBin = "$dest\colmap-3.9.1-windows-cuda"
    [Environment]::SetEnvironmentVariable("Path", "$colmapBin;$env:Path", "User")
    $env:Path = "$colmapBin;$env:Path"
}

# ── Download rgsplat binary ──────────────────────────────────
$target = "x86_64-pc-windows-msvc"
$url = "https://github.com/mfaeezshabbir/rgsplat/releases/latest/download/rgsplat-${target}.zip"
$zip = "$env:TEMP\rgsplat.zip"
$extractDir = "$env:TEMP\rgsplat"

Write-Host "=== Downloading rgsplat ===" -ForegroundColor Cyan
Invoke-WebRequest -Uri $url -OutFile $zip

Remove-Item -Path $extractDir -Recurse -Force -ErrorAction SilentlyContinue
Expand-Archive -Path $zip -DestinationPath $extractDir -Force

# Install to ~\.cargo\bin
$installDir = "$env:USERPROFILE\.cargo\bin"
if (-not (Test-Path $installDir)) { New-Item -ItemType Directory -Path $installDir -Force | Out-Null }
Copy-Item "$extractDir\rgsplat.exe" "$installDir\rgsplat.exe" -Force
Remove-Item -Path $zip -Force -ErrorAction SilentlyContinue
Remove-Item -Path $extractDir -Recurse -Force -ErrorAction SilentlyContinue

# Add to PATH if not already
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($userPath -notlike "*$installDir*") {
    [Environment]::SetEnvironmentVariable("Path", "$installDir;$userPath", "User")
    $env:Path = "$installDir;$env:Path"
}

Write-Host ""
Write-Host "=== Done! Run: rgsplat --help ===" -ForegroundColor Green
