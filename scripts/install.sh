#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────
# rgsplat — one-time install script (Linux / macOS)
# ─────────────────────────────────────────────────────────────
# Usage:  curl -fsSL https://raw.githubusercontent.com/... | bash
# Or:     bash scripts/install.sh
# ─────────────────────────────────────────────────────────────

echo "=== rgsplat: installing system dependencies ==="

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

# ── Platform detection ──────────────────────────────────────
ARCH=""
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    ARCH="x86_64-unknown-linux-gnu"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    if [[ "$(uname -m)" == "arm64" ]]; then
        ARCH="aarch64-apple-darwin"
    else
        ARCH="x86_64-apple-darwin"
    fi
else
    echo "Unsupported OS: $OSTYPE"
    exit 1
fi

# ── Download rgsplat binary ──────────────────────────────────
echo "=== Downloading rgsplat (${ARCH}) ==="
URL="https://github.com/mfaeezshabbir/rgsplat/releases/latest/download/rgsplat-${ARCH}.tar.gz"
TMP_DIR=$(mktemp -d)
curl -fsSL "$URL" -o "$TMP_DIR/rgsplat.tar.gz"
tar xzf "$TMP_DIR/rgsplat.tar.gz" -C "$TMP_DIR"

# Install to ~/.cargo/bin (or /usr/local/bin if we have sudo)
INSTALL_DIR="${HOME}/.cargo/bin"
mkdir -p "$INSTALL_DIR"
cp "$TMP_DIR/rgsplat" "$INSTALL_DIR/rgsplat"
chmod +x "$INSTALL_DIR/rgsplat"
rm -rf "$TMP_DIR"

# Ensure on PATH
if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
    echo "  Add to your shell profile: export PATH=\"\$PATH:$INSTALL_DIR\""
fi

echo ""
echo "=== Done! Run: rgsplat --help ==="
