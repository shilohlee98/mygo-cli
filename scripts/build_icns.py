#!/usr/bin/env python3
"""Build a modern PNG-backed macOS ICNS file without relying on iconutil."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path


ICON_CHUNKS = (
    (b"icp4", "icon_16x16.png"),
    (b"icp5", "icon_32x32.png"),
    (b"icp6", "icon_32x32@2x.png"),
    (b"ic07", "icon_128x128.png"),
    (b"ic08", "icon_128x128@2x.png"),
    (b"ic09", "icon_256x256@2x.png"),
    (b"ic10", "icon_512x512@2x.png"),
)


def build_icns(iconset_dir: Path, output_file: Path) -> None:
    chunks = []
    for chunk_type, file_name in ICON_CHUNKS:
        png = (iconset_dir / file_name).read_bytes()
        if not png.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError(f"Not a PNG file: {file_name}")
        chunks.append(chunk_type + struct.pack(">I", len(png) + 8) + png)

    payload = b"".join(chunks)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_bytes(b"icns" + struct.pack(">I", len(payload) + 8) + payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("iconset_dir", type=Path)
    parser.add_argument("output_file", type=Path)
    args = parser.parse_args()
    build_icns(args.iconset_dir, args.output_file)


if __name__ == "__main__":
    main()
