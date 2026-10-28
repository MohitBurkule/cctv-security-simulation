#!/usr/bin/env bash
# Install Unreal Editor 5.8.2 from the AUR package, given a manually downloaded zip.
#
# The AUR `unreal-engine-bin` package CANNOT fetch the archive itself — its
# PKGBUILD sets a `manual::` DLAGENT precisely because Epic puts the Linux build
# behind an account login. So the one step no script can do is the download:
# you must fetch Linux_Unreal_Engine_5.8.2.zip from
#
#     https://www.unrealengine.com/linux
#
# while signed in, and hand its path to this script. Everything after that is
# automated.
#
# Usage: scripts/install_unreal_editor.sh /path/to/Linux_Unreal_Engine_5.8.2.zip
#
# Expect this to take a while and a lot of disk: the zip alone is tens of GB,
# makepkg extracts it and then builds a package from the result, so peak usage
# is roughly three times the archive size before the final install.

set -euo pipefail

PKGVER=5.8.2
ZIP_NAME="Linux_Unreal_Engine_${PKGVER}.zip"
EXPECTED_SHA=4bc938a3c34ae7305e9dff7b3be381a43ca4b942348e90102885f4bcab7e9575
BUILD_DIR=${UE_BUILD_DIR:-$HOME/data/ue-build}
DEPS=(alsa-lib at-spi2-core avahi glu libxcomposite libxdamage libxkbcommon
      libxrandr lttng-ust2.12 nss pango)

die() { echo "error: $*" >&2; exit 1; }

ZIP=${1:-}
[ -n "$ZIP" ] || die "give me the path to $ZIP_NAME (see the header of this script)"
[ -f "$ZIP" ] || die "no such file: $ZIP"

# --- space check before anything expensive ---------------------------------
zip_bytes=$(stat -c %s "$ZIP")
need_kb=$(( (zip_bytes / 1024) * 3 ))
free_kb=$(df --output=avail -k "$BUILD_DIR" 2>/dev/null | tail -1 || df --output=avail -k "$HOME" | tail -1)
echo "archive: $(( zip_bytes / 1024 / 1024 )) MB   need ~$(( need_kb / 1024 / 1024 )) GB   free: $(( free_kb / 1024 / 1024 )) GB"
[ "$free_kb" -gt "$need_kb" ] || die "not enough free space for extract + package"

# --- verify the archive is the one the PKGBUILD expects --------------------
# A truncated or wrong-version download otherwise fails many minutes into
# makepkg, after the extract, which is an expensive way to find out.
echo "checksumming (this reads the whole archive)..."
actual=$(sha256sum "$ZIP" | cut -d' ' -f1)
if [ "$actual" != "$EXPECTED_SHA" ]; then
  echo "  expected $EXPECTED_SHA" >&2
  echo "  actual   $actual" >&2
  die "checksum mismatch — wrong version, or an incomplete download"
fi
echo "  checksum ok"

# --- dependencies -----------------------------------------------------------
echo "installing dependencies..."
sudo pacman -S --needed --noconfirm "${DEPS[@]}"

# --- build ------------------------------------------------------------------
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
if [ -d unreal-engine-bin/.git ]; then
  git -C unreal-engine-bin pull --ff-only
else
  rm -rf unreal-engine-bin
  git clone --depth 1 https://aur.archlinux.org/unreal-engine-bin.git
fi
cd unreal-engine-bin

# makepkg looks for the archive in the build directory by exact name. Hard-link
# rather than copy when possible: these archives are far too large to duplicate.
if [ ! -f "$ZIP_NAME" ]; then
  ln "$ZIP" "$ZIP_NAME" 2>/dev/null || cp -v "$ZIP" "$ZIP_NAME"
fi

echo "building (long — extraction dominates)..."
makepkg -si --noconfirm --needed

echo
echo "installed to /opt/unreal-engine"
echo "editor binary: /opt/unreal-engine/Engine/Binaries/Linux/UnrealEditor"
"$(dirname "$0")/verify_unreal_editor.sh" || true
