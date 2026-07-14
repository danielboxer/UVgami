#!/usr/bin/env bash
# Extract engine binaries from engine zip files and bundle them into the addon.
# Usage: bundle-engines.sh <addon_dir> <engine_zip_prefix>
# Example: bundle-engines.sh UVgami optcuts-engine

set -euo pipefail

ADDON_DIR="$1"
ENGINE_ZIP_PREFIX="$2"

engine_zips=( ${ENGINE_ZIP_PREFIX}-*.zip )
if [ ! -f "${engine_zips[0]}" ]; then
  echo "No engine zips found matching ${ENGINE_ZIP_PREFIX}-*.zip"
  exit 1
fi

mkdir -p "${ADDON_DIR}/engines"

for zip_file in "${engine_zips[@]}"; do
  # extract platform from filename (e.g., optcuts-engine-1.1.2-windows.zip -> windows)
  platform=$(echo "$zip_file" | sed "s/${ENGINE_ZIP_PREFIX}-[0-9.]*-//" | sed 's/\.zip//')

  tmp_dir=$(mktemp -d)
  unzip -o "$zip_file" -d "$tmp_dir"

  # find the engine binary (named uvgami or uvgami.exe)
  engine_bin=$(find "$tmp_dir" -name "uvgami" -o -name "uvgami.exe" | head -1)
  if [ -n "$engine_bin" ]; then
    mkdir -p "${ADDON_DIR}/engines/${platform}"
    cp "$engine_bin" "${ADDON_DIR}/engines/${platform}/"
    echo "Bundled engine for ${platform}"
  else
    echo "Warning: no engine binary found in ${zip_file}"
  fi

  rm -rf "$tmp_dir"
done

# ship the license notices for the optcuts binary and everything statically
# linked into it, next to the binaries. mit and apache-2.0 require the notice
# travel with the binary; mpl-2.0 requires the notice plus source availability.
license_dir="${ADDON_DIR}/engines/licenses"
mkdir -p "$license_dir"
cp engine/optcuts/LICENSE.txt "$license_dir/OptCuts-LICENSE-MIT.txt"
cp engine/optcuts/ext/libigl/LICENSE.MPL2 "$license_dir/libigl-LICENSE-MPL2.txt"
cp engine/optcuts/ext/tbb/LICENSE "$license_dir/oneTBB-LICENSE-Apache2.txt"
cp engine/optcuts/ext/mimalloc/LICENSE "$license_dir/mimalloc-LICENSE-MIT.txt"

cat > "$license_dir/README.txt" <<'EOF'
The UVgami OptCuts engine binary (engines/<platform>/uvgami[.exe]) statically
links these components. Their license texts are in this folder.

  OptCuts    MIT         OptCuts-LICENSE-MIT.txt
  libigl     MPL-2.0     libigl-LICENSE-MPL2.txt
  Eigen      MPL-2.0     https://www.mozilla.org/MPL/2.0/ (headers, fetched at build)
  oneTBB     Apache-2.0  oneTBB-LICENSE-Apache2.txt
  mimalloc   MIT         mimalloc-LICENSE-MIT.txt

The UVgami add-on itself is GPL-3.0-or-later (see the add-on's LICENSE).
EOF
echo "Bundled engine license notices"
