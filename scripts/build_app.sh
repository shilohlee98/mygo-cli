#!/bin/sh
set -eu

cd "$(dirname "$0")/.."

mygo_uv_cache="$PWD/.uv-cache"
mygo_pyinstaller_cache="$PWD/build/pyinstaller-config"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$mygo_uv_cache}"
export PYINSTALLER_CONFIG_DIR="${PYINSTALLER_CONFIG_DIR:-$mygo_pyinstaller_cache}"

uv run --group app pyinstaller --clean --noconfirm MyGo.spec

mygo_runtime_dir="$PWD/dist/MyGo"
mygo_app_dir="$PWD/dist/MyGo.app"
mygo_contents_dir="$mygo_app_dir/Contents"
mygo_icon_source="$PWD/packaging/MyGoIcon.png"
mygo_iconset_dir="$PWD/build/MyGo.iconset"
mygo_codesign_identity="${MYGO_CODESIGN_IDENTITY:--}"

if [ ! -x "$mygo_runtime_dir/MyGo" ]; then
    printf 'Missing frozen runtime: %s\n' "$mygo_runtime_dir/MyGo" >&2
    exit 1
fi

rm -rf "$mygo_app_dir"
mkdir -p "$mygo_contents_dir/MacOS" "$mygo_contents_dir/Resources"
clang -O2 -mmacosx-version-min=13.0 packaging/MyGoLauncher.c -o "$mygo_contents_dir/MacOS/MyGoLauncher"
install -m 644 packaging/Info.plist "$mygo_contents_dir/Info.plist"

if [ ! -f "$mygo_icon_source" ]; then
    printf 'Missing app icon source: %s\n' "$mygo_icon_source" >&2
    exit 1
fi

rm -rf "$mygo_iconset_dir"
mkdir -p "$mygo_iconset_dir"
sips -z 16 16 "$mygo_icon_source" --out "$mygo_iconset_dir/icon_16x16.png" >/dev/null
sips -z 32 32 "$mygo_icon_source" --out "$mygo_iconset_dir/icon_16x16@2x.png" >/dev/null
sips -z 32 32 "$mygo_icon_source" --out "$mygo_iconset_dir/icon_32x32.png" >/dev/null
sips -z 64 64 "$mygo_icon_source" --out "$mygo_iconset_dir/icon_32x32@2x.png" >/dev/null
sips -z 128 128 "$mygo_icon_source" --out "$mygo_iconset_dir/icon_128x128.png" >/dev/null
sips -z 256 256 "$mygo_icon_source" --out "$mygo_iconset_dir/icon_128x128@2x.png" >/dev/null
sips -z 256 256 "$mygo_icon_source" --out "$mygo_iconset_dir/icon_256x256.png" >/dev/null
sips -z 512 512 "$mygo_icon_source" --out "$mygo_iconset_dir/icon_256x256@2x.png" >/dev/null
sips -z 512 512 "$mygo_icon_source" --out "$mygo_iconset_dir/icon_512x512.png" >/dev/null
sips -z 1024 1024 "$mygo_icon_source" --out "$mygo_iconset_dir/icon_512x512@2x.png" >/dev/null
uv run python scripts/build_icns.py \
    "$mygo_iconset_dir" \
    "$mygo_contents_dir/Resources/MyGo.icns"

mv "$mygo_runtime_dir" "$mygo_contents_dir/Resources/MyGoRuntime"
if [ "$mygo_codesign_identity" = "-" ]; then
    codesign --force --deep --sign - "$mygo_app_dir"
else
    codesign \
        --force \
        --deep \
        --options runtime \
        --timestamp \
        --sign "$mygo_codesign_identity" \
        "$mygo_app_dir"
fi
codesign --verify --deep --strict "$mygo_app_dir"

printf '\nBuilt: %s/dist/MyGo.app\n' "$PWD"
