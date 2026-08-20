#!/bin/sh
set -eu

cd "$(dirname "$0")/.."

./scripts/build_app.sh

mygo_app_dir="$PWD/dist/MyGo.app"
mygo_version="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' packaging/Info.plist)"
mygo_arch="$(uname -m)"
mygo_dmg_root="$PWD/build/dmg-root"
mygo_dmg_file="$PWD/dist/MyGo-${mygo_version}-${mygo_arch}.dmg"
mygo_codesign_identity="${MYGO_CODESIGN_IDENTITY:--}"

rm -rf "$mygo_dmg_root"
mkdir -p "$mygo_dmg_root"
ditto "$mygo_app_dir" "$mygo_dmg_root/MyGo.app"
ln -s /Applications "$mygo_dmg_root/Applications"

hdiutil create \
    -volname "MyGo ${mygo_version}" \
    -srcfolder "$mygo_dmg_root" \
    -format UDZO \
    -ov \
    "$mygo_dmg_file"

if [ "$mygo_codesign_identity" != "-" ]; then
    codesign \
        --force \
        --timestamp \
        --identifier com.shilohlee.mygo.dmg \
        --sign "$mygo_codesign_identity" \
        "$mygo_dmg_file"
fi

shasum -a 256 "$mygo_dmg_file" > "$mygo_dmg_file.sha256"

printf '\nBuilt: %s\n' "$mygo_dmg_file"
printf 'SHA-256: %s.sha256\n' "$mygo_dmg_file"
