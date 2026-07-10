"""Download a size-spanning set of test OBJs from alecjacobson/common-3d-test-models.

Stdlib only. Names are validated against the repo's live data/ listing rather
than guessed, then fetched by raw URL.
"""

import json
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"
LISTING_URL = (
    "https://api.github.com/repos/alecjacobson/common-3d-test-models/contents/data"
)
RAW_BASE = (
    "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data"
)

# curated to span roughly 5k to 120k+ triangles; actual counts printed after
CURATED = [
    "spot",
    "suzanne",
    "fandisk",
    "cow",
    "cheburashka",
    "homer",
    "beast",
    "armadillo",
    "max-planck",
    "ogre",
    "nefertiti",
]


def available_names():
    request = urllib.request.Request(LISTING_URL, headers={"User-Agent": "uvgami"})
    with urllib.request.urlopen(request) as response:
        listing = json.load(response)
    return {entry["name"] for entry in listing if entry["name"].endswith(".obj")}


def triangle_count(path):
    count = 0
    with open(path) as file:
        for line in file:
            if line.startswith("f "):
                count += len(line.split()) - 3  # fan-triangulated polygon
    return count


def download(names=None):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    present = available_names()
    wanted = names or CURATED
    missing = [n for n in wanted if f"{n}.obj" not in present]
    if missing:
        print(f"not in repo, skipping: {', '.join(missing)}", file=sys.stderr)

    paths = []
    for name in wanted:
        if f"{name}.obj" not in present:
            continue
        dest = MODELS_DIR / f"{name}.obj"
        if dest.exists():
            print(f"have {name}.obj")
        else:
            print(f"downloading {name}.obj")
            request = urllib.request.Request(
                f"{RAW_BASE}/{name}.obj", headers={"User-Agent": "uvgami"}
            )
            with urllib.request.urlopen(request) as response:
                dest.write_bytes(response.read())
        paths.append(dest)

    print("\nmodel triangle counts:")
    for path in sorted(paths, key=triangle_count):
        print(f"  {path.stem:<16} {triangle_count(path):>9,} tris")
    return paths


if __name__ == "__main__":
    download()
