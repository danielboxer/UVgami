"""Turn a staged full-addon directory into per-platform extension zips for
extensions.blender.org, which bans runtime downloads. Each zip drops the partuv
engine (its installer is the only network use) and keeps a single platform's
optcuts binary, with the manifest's network permission removed and a platforms
tag added.

Stdlib only, runs on any platform.
"""

import argparse
import re
import sys
import zipfile
from pathlib import Path

# engines/<dir> -> blender_manifest.toml platform tag
PLATFORM_TAGS = {
    "windows": "windows-x64",
    "linux": "linux-x64",
    "macos-arm64": "macos-arm64",
    "macos-x64": "macos-x64",
}

PARTUV_PKG = ("src", "engines", "partuv")


def fail(message):
    print(f"error: {message}", file=sys.stderr)
    sys.exit(1)


def read_version(manifest_text):
    match = re.search(r'^version\s*=\s*"([^"]+)"', manifest_text, re.MULTILINE)
    if not match:
        fail("blender_manifest.toml has no version line")
    return match.group(1)


def transform_manifest(manifest_text, tag):
    """Drop the network permission and add a platforms tag, leaving the rest
    byte-identical."""
    out = []
    inserted = False
    for line in manifest_text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("network") and "=" in stripped:
            # the only network use is the deleted partuv installer
            continue
        out.append(line)
        if not inserted and stripped.startswith("website"):
            out.append(f'platforms = ["{tag}"]\n')
            inserted = True
    if not inserted:
        fail("blender_manifest.toml has no website line to anchor platforms after")
    return "".join(out)


def is_excluded(parts, platform_dir):
    if "__pycache__" in parts:
        return True
    if tuple(parts[:3]) == PARTUV_PKG:
        return True
    # keep only this platform's engine dir plus the shared licenses
    if parts[0] == "engines" and len(parts) >= 2:
        if parts[1] not in ("licenses", platform_dir):
            return True
    return False


def build_zip(staged, out_dir, platform_dir, version, manifest_text):
    tag = PLATFORM_TAGS[platform_dir]
    zip_path = out_dir / f"uvgami-{version}-{tag}.zip"
    manifest_rel = "blender_manifest.toml"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(staged.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(staged)
            if is_excluded(rel.parts, platform_dir):
                continue
            arcname = rel.as_posix()
            if arcname == manifest_rel:
                archive.writestr(arcname, transform_manifest(manifest_text, tag))
            else:
                archive.write(path, arcname)
    return zip_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("staged_addon_dir", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    staged = args.staged_addon_dir
    if not (staged / PARTUV_PKG[0] / PARTUV_PKG[1] / PARTUV_PKG[2]).is_dir():
        fail("src/engines/partuv missing in the staged dir; stage layout drifted")

    engines_dir = staged / "engines"
    platform_dirs = (
        sorted(
            p.name
            for p in engines_dir.iterdir()
            if p.is_dir() and p.name in PLATFORM_TAGS
        )
        if engines_dir.is_dir()
        else []
    )
    if not platform_dirs:
        fail("no engines/<platform> dirs found in the staged dir")

    manifest_text = (staged / "blender_manifest.toml").read_text(encoding="utf-8")
    version = read_version(manifest_text)

    args.out.mkdir(parents=True, exist_ok=True)
    for platform_dir in platform_dirs:
        zip_path = build_zip(staged, args.out, platform_dir, version, manifest_text)
        print(f"wrote {zip_path}")


if __name__ == "__main__":
    main()
