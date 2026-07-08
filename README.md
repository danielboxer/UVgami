# UVgami

![UVgami](https://user-images.githubusercontent.com/65575771/211176523-4b9f7aa1-0994-4c54-928f-e0ec319052d9.gif)

## Quickstart

> [!NOTE]
> Check the [user guide](/docs/docs.md) for more detailed documentation.

[Find the latest release here](https://github.com/DanielBoxer/UVgami/releases/latest) and pick one install option below:

### Option 1: Download the engine separately

1. Get the add-on: `UVgami.zip`
2. Get the correct engine for your operating system: (`uvgami-engine-X.X.X-operating-system.zip`)
3. Set the engine path in the add-on preferences after installing

### Option 2: Download the engine bundled with the add-on

1. Get the bundle: `UVgami-bundled-with-engines.zip`
2. The add-on will auto detect the engine since it's bundled

![Elephant](https://github.com/DanielBoxer/UVgami/assets/65575771/07dd8351-5acb-493d-b35e-422d8da35a7f)

![Elephant 2](https://github.com/DanielBoxer/UVgami/assets/65575771/bc33389b-b902-4334-85ec-fa637dce0fbb)

![Seam Restrictions](https://github.com/DanielBoxer/UVgami/assets/65575771/632fb52f-b9fc-4b98-a3ed-c8b679fe3102)

![Ostrich](https://github.com/DanielBoxer/UVgami/assets/65575771/44f181df-7d06-4b35-aa32-3ef380a5c986)

![Rhino](https://github.com/DanielBoxer/UVgami/assets/65575771/12b691fc-4ff6-4462-9dbc-b615a85bf7fc)

The unwrapping engine is from [Optcuts](https://github.com/liminchen/OptCuts) by Minchen Li, licensed under the MIT License, and has been modified to work in Blender.

## Developer CLI

Blender-independent CLI for testing the engines with OBJ files. Needs [uv](https://docs.astral.sh/uv/).

```powershell
uv sync
uv run uvgami unwrap model.obj --engine optcuts
```

- Output defaults to `<input stem>_uv.obj` next to the input, use `-o` and `--overwrite` to control it
- `--json` prints one machine-readable result on stdout, all logs go to stderr
- OptCuts options: `--quality`, `--seam-weight`, `--seam-weights`, `--import-uvs`, `--optcuts-path` (defaults to the bundled `engines/` binary)
- Exit codes: 0 ok, 2 invalid input, 3 missing runtime files, 4 engine failure, 5 bad output

### PartUV engine

PartUV needs CUDA. It builds natively on Windows and Linux; on Windows without a native install the CLI bridges to WSL by itself: the same `unwrap` command re-invokes the CLI inside the distro with paths translated. `UVGAMI_PARTUV_WSL=1` forces the bridge even when the native build exists.

In the add-on, PartUV runs through this CLI: from a repo checkout it uses `uv run`, otherwise the install button in the add-on preferences downloads the wheel from the latest release and installs it with Blender's Python (`python -m pip install --target`), and unwraps run as `python -m uvgami_cli`.

Two segmentation modes drive the part tree:

- `--segmentation ai` (default): PartField inference, best quality; needs the torch stack (`--extra partuv`) and the [PartField checkpoint](https://huggingface.co/mikaelaangel/partfield-ckpt) (untracked)
- `--segmentation geometric`: scikit-learn agglomerative clustering on face normals and centroids; no torch, no checkpoint, installs with `--extra partuv-lite`

One-time Windows setup:

1. Visual Studio 2026 with the C++ workload (includes vcpkg, cmake, ninja)
2. CUDA Toolkit 13.2+ (`winget install Nvidia.CUDA`), older versions don't support VS 2026
3. From a VS dev shell (`Launch-VsDevShell.ps1 -Arch amd64`): `uv sync --extra partuv` (or `--extra partuv-lite`)

The vcpkg deps (cgal, yaml-cpp, tbb) build from source on the first run, about 50 minutes, and are binary-cached after. The wheel bundles the runtime DLLs; running it only needs the VC++ redistributable.

One-time WSL setup (Ubuntu 24.04):

```bash
sudo apt install build-essential libcgal-dev libyaml-cpp-dev libtbb-dev
# cuda toolkit from the nvidia wsl-ubuntu repo, then nvcc is at /usr/local/cuda-12.6/bin
sudo apt install cuda-toolkit-12-6
wget -P engine/partuv https://huggingface.co/mikaelaangel/partfield-ckpt/resolve/main/model_objaverse.ckpt
```

Build and run in WSL (from the repo):

```bash
export PATH=/usr/local/cuda-12.6/bin:$PATH
# venv on ext4: keeps the Windows .venv untouched and torch imports fast
export UV_PROJECT_ENVIRONMENT=~/uvgami-venv
uv sync --extra partuv        # or --extra partuv-lite for geometric only
uv run uvgami unwrap model.obj --engine partuv
```

After that, the same unwrap works from Windows directly (PowerShell, not Git Bash: MSYS mangles POSIX paths in env vars).

- Checkpoint lookup order: `--checkpoint`, `$UVGAMI_PARTUV_CHECKPOINT`, `engine/partuv/model_objaverse.ckpt`. A WSL-side path like `/root/model.ckpt` passes through the bridge untranslated
- Bridge env vars: `UVGAMI_WSL_DISTRO` (default: first non-Docker distro), `UVGAMI_WSL_VENV` (default: `~/uvgami-venv` in the distro)
- The extension compiles to `/var/tmp/partuv-build` in WSL (compiling on `/mnt/c` is hopelessly slow) and targets sm_86 (RTX 3060) by default, override with the `CUDAARCHS` env var; `-DPARTUV_NATIVE=ON` restores upstream's `-march=native`
- Release wheels should widen the CUDA targets: `CUDAARCHS="75-real;80-real;86-real;89-real;90-real;120"` with CUDA 13 (Windows; sm_75 is its floor), drop `120` on CUDA 12.6 (WSL; sm_90 is its ceiling)
- `--threshold` sets the distortion threshold (default 1.25), `--config` overrides `engine/partuv/config/config.yaml`

### Tests

```powershell
uv run pytest              # unit tests + OptCuts smoke test
uv run pytest -m "not smoke"
uv run ruff check uvgami_cli tests
```
