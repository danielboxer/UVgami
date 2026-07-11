"""Benchmark the optcuts and partuv engines over the models in bench/models/.

Runs each engine once in batch mode, timestamps the per-mesh start/done/failed
markers on stdout for wall time, then scores every output OBJ with metrics.py.
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import metrics

BENCH = Path(__file__).parent
REPO_ROOT = BENCH.parent
MODELS_DIR = BENCH / "models"
OUT_DIR = BENCH / "out"
RESULTS_CSV = BENCH / "results.csv"
ENGINES = ["optcuts", "partuv"]
WARMUP_STEM = "_warmup"
# optcuts is too slow past its cap (triangle count); partuv is uncapped.
# --max-faces overrides both with one cap for all engines.
ENGINE_FACE_CAPS = {"optcuts": 13_500, "partuv": None}
# --quick caps both engines here so a run finishes in a few minutes
QUICK_MAX_FACES = 6_000

METRIC_FIELDS = [
    "chart_count",
    "seam_length",
    "area_distortion",
    "angle_distortion",
    "uv_utilization",
    "degenerate_tris",
]


def smallest(paths):
    return min(paths, key=lambda p: p.stat().st_size)


def scan_obj(path):
    """Return (tri_count, has_ngon). tri_count is the fan-triangulated triangle
    count (len(face verts) - 2 per face) used for the per-engine face caps.
    has_ngon flags any face with more than 3 vertices, so it needs
    triangulating before an engine sees it."""
    tris = 0
    has_ngon = False
    with open(path) as file:
        for line in file:
            if line.startswith("f "):
                verts = len(line.split()) - 1
                tris += verts - 2
                if verts > 3:
                    has_ngon = True
    return tris, has_ngon


def triangulate_obj(src, dest):
    """Fan-triangulate every face of src into dest: verts v0..vn become
    triangles (v0, vi, vi+1). Full vertex tokens and all non-face lines are
    kept verbatim."""
    with open(src) as fin, open(dest, "w") as fout:
        for line in fin:
            if line.startswith("f "):
                verts = line.split()[1:]
                if len(verts) > 3:
                    for i in range(1, len(verts) - 1):
                        fout.write(f"f {verts[0]} {verts[i]} {verts[i + 1]}\n")
                    continue
            fout.write(line)


def run_engine(engine, model_paths, warmup_src, warmup_dir):
    """Invoke the CLI once over all models plus a leading warmup mesh, returning
    {stem: {"status", "seconds"}}. The warmup absorbs one-time engine cost (for
    partuv the CUDA context and first-kernel setup that lands in mesh one's
    window) and is dropped from the results."""
    out_dir = OUT_DIR / engine
    out_dir.mkdir(parents=True, exist_ok=True)

    warmup = warmup_dir / f"{WARMUP_STEM}.obj"
    shutil.copyfile(warmup_src, warmup)
    inputs = [warmup, *model_paths]

    cmd = [
        "uv",
        "run",
        "--no-sync",
        "uvgami",
        "unwrap",
        *[str(p) for p in inputs],
        "--engine",
        engine,
        "--output-dir",
        str(out_dir),
        "--overwrite",
    ]
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    stderr_log = out_dir / "stderr.log"

    results = {}
    starts = {}
    print(f"\n=== {engine} ===", flush=True)
    with open(stderr_log, "w", encoding="utf-8") as errfile:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=errfile,
            text=True,
            bufsize=1,
            env=env,
        )
        for line in process.stdout:
            now = time.perf_counter()
            line = line.strip()
            if line.startswith("start:"):
                stem = line.split(maxsplit=1)[1]
                starts[stem] = now
                print(f"  start {stem}", flush=True)
            elif line.startswith("done:"):
                stem = line.split(maxsplit=1)[1]
                seconds = now - starts.get(stem, now)
                results[stem] = {"status": "ok", "seconds": seconds}
                print(f"  done  {stem}  {seconds:.1f}s", flush=True)
            elif line.startswith("failed:"):
                stem = line.split(maxsplit=1)[1].split()[0]
                seconds = now - starts.get(stem, now)
                results[stem] = {"status": "failed", "seconds": seconds}
                print(f"  FAIL  {stem}", flush=True)
        code = process.wait()

    for path in model_paths:
        results.setdefault(path.stem, {"status": "no_output", "seconds": None})
    results.pop(WARMUP_STEM, None)
    if code != 0 and all(r["status"] != "ok" for r in results.values()):
        tail = _stderr_tail(stderr_log)
        print(f"  {engine} exited {code}; stderr tail:\n{tail}", file=sys.stderr)
    return results


def _stderr_tail(path, lines=20):
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(text[-lines:])


def score(engine, results):
    out_dir = OUT_DIR / engine
    rows = []
    for stem, info in sorted(results.items()):
        row = {
            "mesh": stem,
            "engine": engine,
            "seconds": info["seconds"],
            "status": info["status"],
        }
        output = out_dir / f"{stem}.obj"
        if info["status"] == "ok" and output.is_file():
            row.update(metrics.compute_metrics(output))
        else:
            row.update({field: None for field in METRIC_FIELDS})
        rows.append(row)
    return rows


def write_csv(rows):
    fields = ["mesh", "engine", "seconds", "status", *METRIC_FIELDS]
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def print_report(rows, engines):
    meshes = sorted({row["mesh"] for row in rows})
    by_key = {(row["mesh"], row["engine"]): row for row in rows}
    cols = [
        "seconds",
        "chart_count",
        "seam_length",
        "area_distortion",
        "angle_distortion",
    ]
    print("\nper-mesh comparison:")
    header = f"{'mesh':<16}{'engine':<10}" + "".join(f"{c:>16}" for c in cols)
    print(header)
    print("-" * len(header))
    for mesh in meshes:
        for engine in engines:
            row = by_key.get((mesh, engine))
            if not row:
                continue
            line = f"{mesh:<16}{engine:<10}" + "".join(
                f"{_fmt(row[c]):>16}" for c in cols
            )
            print(line)

    ok_meshes = {
        engine: {
            r["mesh"] for r in rows if r["engine"] == engine and r["status"] == "ok"
        }
        for engine in engines
    }
    common = set.intersection(*ok_meshes.values()) if ok_meshes else set()

    print("\nper-engine means (meshes ok in all engines):")
    if not common:
        print("  (no mesh is ok in every engine, means omitted)")
    for engine in engines:
        ok = [r for r in rows if r["engine"] == engine and r["status"] == "ok"]
        n = len(ok)
        print(
            f"  {engine}: {n} ok, {sum(1 for r in rows if r['engine'] == engine and r['status'] != 'ok')} not ok"
        )
        shared = [r for r in ok if r["mesh"] in common]
        if not shared:
            continue
        for field in ["seconds", *METRIC_FIELDS]:
            values = [r[field] for r in shared if r[field] is not None]
            if values:
                print(f"    {field:<18} {sum(values) / len(values):.3f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="benchmark UV unwrap engines")
    parser.add_argument("--engine", choices=ENGINES, help="run just one engine")
    parser.add_argument(
        "--models", default="*", help="glob over bench/models stems, default all"
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=None,
        help="cap triangle count for every engine, overrides the per-engine defaults",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=f"fast run: cap every engine at {QUICK_MAX_FACES} tris",
    )
    args = parser.parse_args(argv)
    if args.quick:
        if args.max_faces is not None:
            parser.error("--quick and --max-faces are mutually exclusive")
        args.max_faces = QUICK_MAX_FACES

    model_paths = sorted(MODELS_DIR.glob(f"{args.models}.obj"))
    if not model_paths:
        parser.error(f"no models matched {args.models!r} in {MODELS_DIR}")
    engines = [args.engine] if args.engine else ENGINES

    all_rows = []
    with tempfile.TemporaryDirectory(prefix="uvgami-bench-") as tmp:
        tmp = Path(tmp)
        # triangulate ngon meshes into tmp; both engines get identical tri-only
        # input. tri-only sources are passed through untouched.
        prepared = {}
        tri_counts = {}
        for path in model_paths:
            tris, has_ngon = scan_obj(path)
            tri_counts[path.stem] = tris
            if has_ngon:
                dest = tmp / path.name
                triangulate_obj(path, dest)
                prepared[path.stem] = dest
            else:
                prepared[path.stem] = path
        prepared_paths = [prepared[p.stem] for p in model_paths]

        # every prepared input is tri-only, so the smallest is a valid warmup
        warmup_src = smallest(prepared_paths)
        print(f"models: {', '.join(p.stem for p in model_paths)}")

        for engine in engines:
            cap = (
                args.max_faces
                if args.max_faces is not None
                else ENGINE_FACE_CAPS[engine]
            )
            kept = model_paths
            if cap is not None:
                too_big = [p for p in model_paths if tri_counts[p.stem] > cap]
                if too_big:
                    print(
                        f"{engine}: skipped over {cap} tris: "
                        f"{', '.join(p.stem for p in too_big)}"
                    )
                kept = [p for p in model_paths if tri_counts[p.stem] <= cap]
            if not kept:
                print(f"{engine}: all models over cap, skipping engine")
                continue
            engine_paths = [prepared[p.stem] for p in kept]
            results = run_engine(engine, engine_paths, warmup_src, tmp)
            all_rows.extend(score(engine, results))

    write_csv(all_rows)
    print_report(all_rows, engines)
    print(f"\nwrote {RESULTS_CSV}")


if __name__ == "__main__":
    main()
