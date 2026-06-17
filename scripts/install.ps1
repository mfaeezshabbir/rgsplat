# ─────────────────────────────────────────────────────────────
# rgsplat — one-time install script (Windows)
# ─────────────────────────────────────────────────────────────
# Usage:  powershell -ExecutionPolicy Bypass -File install.ps1
# ─────────────────────────────────────────────────────────────

$ErrorActionPreference = "Stop"
$RepoDir = Split-Path -Parent $PSScriptRoot

Write-Host "=== rgsplat: installing dependencies ===" -ForegroundColor Cyan

# ── Rust via rustup ─────────────────────────────────────────
if (Get-Command rustc -ErrorAction SilentlyContinue) {
    Write-Host "  Rust OK ($(rustc --version))" -ForegroundColor Green
} else {
    Write-Host "  Installing Rust via rustup..." -ForegroundColor Yellow
    $url = "https://static.rust-lang.org/rustup/dist/x86_64-pc-windows-msvc/rustup-init.exe"
    $tmp = "$env:TEMP\rustup-init.exe"
    Invoke-WebRequest -Uri $url -OutFile $tmp
    & $tmp -y
    # Update PATH for this session
    $env:Path = "$env:USERPROFILE\.cargo\bin;$env:Path"
}

# ── ffmpeg (winget) ──────────────────────────────────────────
if (Get-Command ffmpeg -ErrorAction SilentlyContinue) {
    Write-Host "  ffmpeg OK" -ForegroundColor Green
} else {
    Write-Host "  Installing ffmpeg via winget..." -ForegroundColor Yellow
    winget install --silent --accept-package-agreements FFmpeg
    # Refresh PATH
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
    # Add to PATH
    $colmapBin = "$dest\colmap-3.9.1-windows-cuda"
    [Environment]::SetEnvironmentVariable("Path", "$colmapBin;$env:Path", "User")
    $env:Path = "$colmapBin;$env:Path"
}

# ── Build & install rgsplat ─────────────────────────────────
Write-Host "=== Building rgsplat (this takes a minute) ===" -ForegroundColor Cyan
cargo install --path "$RepoDir" --locked

Write-Host ""
Write-Host "=== Done! Run: rgsplat --help ===" -ForegroundColor Green
