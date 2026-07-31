#!/usr/bin/env bash
# Extract engine binaries from engine zip files and bundle them into the addon.
# Usage: bundle-engines.sh <addon_dir> <engine_zip_prefix>

set -euo pipefail

ADDON_DIR="$1"
ENGINE_ZIP_PREFIX="$2"

# the binary name is the prefix without the -engine suffix (optcuts, xatlas)
engine_bin_name="${ENGINE_ZIP_PREFIX%-engine}"

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

  # find the engine binary (named e.g. optcuts or optcuts.exe)
  engine_bin=$(find "$tmp_dir" -name "$engine_bin_name" -o -name "${engine_bin_name}.exe" | head -1)
  if [ -n "$engine_bin" ]; then
    mkdir -p "${ADDON_DIR}/engines/${platform}"
    cp "$engine_bin" "${ADDON_DIR}/engines/${platform}/"
    echo "Bundled engine for ${platform}"
  else
    echo "Warning: no engine binary found in ${zip_file}"
  fi

  rm -rf "$tmp_dir"
done

license_dir="${ADDON_DIR}/engines/licenses"
mkdir -p "$license_dir"
# the engine notice is short form, the gpl body it refers to is the addon's
{ cat engine/optcuts/LICENSE.txt; echo; cat LICENSE; } > "$license_dir/OptCuts-LICENSE-GPL3.txt"
cp engine/optcuts/ext/libigl/LICENSE.MPL2 "$license_dir/libigl-LICENSE-MPL2.txt"
cp engine/optcuts/src/include/tclap/COPYING "$license_dir/tclap-LICENSE-MIT.txt"
# tbb and mimalloc are downloaded at build time into gitignored ext/ dirs, so
# their notices are vendored in engine/licenses. that stays outside
# engine/optcuts because the engine build treats any diff there as needing a
# version bump.
cp engine/licenses/oneTBB-LICENSE-Apache2.txt "$license_dir/"
cp engine/licenses/mimalloc-LICENSE-MIT.txt "$license_dir/"
cp engine/licenses/README.txt "$license_dir/"
# xatlas is dependency-free and links nothing extra
cp engine/xatlas/LICENSE "$license_dir/xatlas-LICENSE-MIT.txt"
echo "Bundled engine license notices"
