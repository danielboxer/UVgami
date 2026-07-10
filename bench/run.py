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


def run_engine(engine, model_paths, warmup_dir):
    """Invoke the CLI once over all models plus a leading warmup mesh, returning
    {stem: {"status", "seconds"}}. The warmup absorbs one-time engine cost (for
    partuv the CUDA context and first-kernel setup that lands in mesh one's
    window) and is dropped from the results."""
    out_dir = OUT_DIR / engine
    out_dir.mkdir(parents=True, exist_ok=True)

    warmup = warmup_dir / f"{WARMUP_STEM}.obj"
    shutil.copyfile(smallest(model_paths), warmup)
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

    print("\nper-engine means (ok meshes only):")
    for engine in engines:
        ok = [r for r in rows if r["engine"] == engine and r["status"] == "ok"]
        n = len(ok)
        print(
            f"  {engine}: {n} ok, {sum(1 for r in rows if r['engine'] == engine and r['status'] != 'ok')} not ok"
        )
        if not ok:
            continue
        for field in ["seconds", *METRIC_FIELDS]:
            values = [r[field] for r in ok if r[field] is not None]
            if values:
                print(f"    {field:<18} {sum(values) / len(values):.3f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="benchmark UV unwrap engines")
    parser.add_argument("--engine", choices=ENGINES, help="run just one engine")
    parser.add_argument(
        "--models", default="*", help="glob over bench/models stems, default all"
    )
    args = parser.parse_args(argv)

    model_paths = sorted(MODELS_DIR.glob(f"{args.models}.obj"))
    if not model_paths:
        parser.error(f"no models matched {args.models!r} in {MODELS_DIR}")
    engines = [args.engine] if args.engine else ENGINES
    print(f"models: {', '.join(p.stem for p in model_paths)}")

    all_rows = []
    with tempfile.TemporaryDirectory(prefix="uvgami-bench-") as warmup_dir:
        for engine in engines:
            results = run_engine(engine, model_paths, Path(warmup_dir))
            all_rows.extend(score(engine, results))

    write_csv(all_rows)
    print_report(all_rows, engines)
    print(f"\nwrote {RESULTS_CSV}")


if __name__ == "__main__":
    main()
