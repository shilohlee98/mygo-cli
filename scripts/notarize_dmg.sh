#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
    printf 'Usage: MYGO_NOTARY_PROFILE=<profile> %s <dmg-path>\n' "$0" >&2
    exit 2
fi

mygo_dmg_file="$1"
mygo_notary_profile="${MYGO_NOTARY_PROFILE:-}"

if [ -z "$mygo_notary_profile" ]; then
    printf 'MYGO_NOTARY_PROFILE is required.\n' >&2
    exit 2
fi

if [ ! -f "$mygo_dmg_file" ]; then
    printf 'DMG not found: %s\n' "$mygo_dmg_file" >&2
    exit 2
fi

codesign --verify --verbose=2 "$mygo_dmg_file"
xcrun notarytool submit \
    "$mygo_dmg_file" \
    --keychain-profile "$mygo_notary_profile" \
    --wait
xcrun stapler staple "$mygo_dmg_file"
xcrun stapler validate "$mygo_dmg_file"
shasum -a 256 "$mygo_dmg_file" > "$mygo_dmg_file.sha256"

printf '\nNotarized: %s\n' "$mygo_dmg_file"
