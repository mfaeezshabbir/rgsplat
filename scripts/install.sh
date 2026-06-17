#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────
# rgsplat — one-time install script (Linux / macOS)
# ─────────────────────────────────────────────────────────────
# Usage:  curl -fsSL https://raw.githubusercontent.com/... | bash
# Or:     bash scripts/install.sh
# ─────────────────────────────────────────────────────────────

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== rgsplat: installing system dependencies ==="

# ── Rust ────────────────────────────────────────────────────
if command -v rustc &>/dev/null; then
    echo "  Rust OK ($(rustc --version))"
else
    echo "  Installing Rust via rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    . "$HOME/.cargo/env"
fi

# ── System packages (ffmpeg + colmap) ────────────────────────
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if command -v apt-get &>/dev/null; then
        sudo apt-get update -qq
        sudo apt-get install -y -qq ffmpeg colmap
    elif command -v dnf &>/dev/null; then
        sudo dnf install -y ffmpeg colmap
    elif command -v pacman &>/dev/null; then
        sudo pacman -S --noconfirm ffmpeg colmap
    else
        echo "WARNING: unknown package manager — install ffmpeg and COLMAP manually"
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    if command -v brew &>/dev/null; then
        brew install ffmpeg colmap
    else
        echo "WARNING: Homebrew not found — install ffmpeg and COLMAP manually"
    fi
fi

# Verify
for cmd in ffmpeg colmap; do
    if command -v $cmd &>/dev/null; then
        echo "  $cmd OK ($($cmd --version 2>&1 | head -1))"
    else
        echo "  WARNING: $cmd not found — install manually"
    fi
done

# ── Build & install rgsplat ─────────────────────────────────
echo "=== Building rgsplat (this takes a minute) ==="
cargo install --path "$REPO_DIR" --locked

echo ""
echo "=== Done! Run: rgsplat --help ==="
