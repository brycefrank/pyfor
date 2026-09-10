#!/usr/bin/env python3
"""Cross tool benchmark: pyfor, lidR, and PDAL on a single tile.

Each tool runs its own idiomatic pipeline for the same operations:

    read             read the tile into memory
    readwrite        read the tile and write it back out as .laz
    normalize        height above ground (pyfor: Zhang 2003, lidR: tin, PDAL: smrf + hag_nn)
    chm              max z per 1 m cell, written to GeoTIFF
    chm_normalized   normalize, then max height per 1 m cell
    metrics          normalize, then area based metrics per 20 m cell

Every measurement is the wall clock of a fresh process. A warmup run is discarded, the page cache
is warm for all measurements, and tools are pinned to one thread where they allow it, so the numbers
compare single threaded throughput rather than each tool's parallelism defaults.

The algorithms are not identical, see docs/topics/benchmarks.rst for what that means for reading
the results. The CHM operations are directly comparable and the harness verifies that the three
tools produce the same raster before reporting.

Usage:
    python benchmarks/bench.py                 # all modes, all available tools, 3 runs each
    python benchmarks/bench.py --runs 5
    python benchmarks/bench.py --modes chm read --tools pyfor pdal
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "benchmarks"
RESULTS = BENCH / "results"
TILE = ROOT / "pyfortest/data/test.laz"

MODES = ["read", "readwrite", "normalize", "chm", "chm_normalized", "metrics"]
TOOLS = ["pyfor", "lidR", "pdal"]

# PDAL has no area based metrics stage: it is a point cloud translation library, and the per cell
# summaries that pyfor and lidR compute would have to be written by hand as a custom stage.
TOOL_MODES = {
    "pyfor": MODES,
    "lidR": MODES,
    "pdal": [mode for mode in MODES if mode != "metrics"],
}

# The extension the operation writes, chosen so each tool picks the right format for its output.
OUTPUT_EXTENSIONS = {
    "readwrite": ".laz",
    "chm": ".tif",
    "chm_normalized": ".tif",
    "metrics": ".tif",
}


def available_tools(pdal: str | None, rscript: str | None) -> dict[str, str]:
    """The tools that can actually be run on this machine, with the command to run them."""
    tools = {"pyfor": sys.executable}
    if rscript:
        probe = subprocess.run(
            [rscript, "-e",
             'quit(status = !isTRUE(suppressWarnings(requireNamespace("lidR"))))'],
            capture_output=True,
        )
        if probe.returncode == 0:
            tools["lidR"] = rscript
    if pdal:
        probe = subprocess.run([pdal, "--version"], capture_output=True)
        if probe.returncode == 0:
            tools["pdal"] = pdal
    return tools


def pyfor_grid(tile: Path = TILE, cell_size: float = 1.0):
    """The raster grid pyfor produces for a tile.

    This asks pyfor itself, so the harness pins the other tools to exactly the grid pyfor uses. That
    grid is snapped to the cell size, which is the convention GDAL, PDAL, and lidR use, so the grids
    should agree without pinning anything.
    """
    import laspy
    import pyfor

    with laspy.open(tile) as reader:
        header = reader.header
        min_x, min_y = header.mins[0], header.mins[1]
        max_x, max_y = header.maxs[0], header.maxs[1]

    spec = pyfor.rasterizer.GridSpec.covering(min_x, min_y, max_x, max_y, cell_size)
    return {
        "min_x": spec.origin_x,
        "min_y": spec.origin_y,
        "max_x": spec.bounds[2],
        "max_y": spec.bounds[3],
        "n": spec.n,
        "m": spec.m,
        "res": spec.cell_size,
    }


def effective_pipeline(mode: str, threads: int, out_path: Path, grid, tile: Path = TILE):
    """Loads benchmarks/pdal_<mode>.json, rewrites its paths, threads and grid, and writes it out again.

    The pipeline file in the repository is the readable, hand runnable version. This writes the exact
    pipeline that was executed next to the results, so a number can always be traced back to the
    configuration that produced it.
    """
    source = BENCH / f"pdal_{mode}.json"
    pipeline = json.loads(source.read_text())
    for stage in pipeline["pipeline"]:
        if stage["type"] == "readers.las":
            stage["filename"] = str(tile.relative_to(ROOT)) if tile.is_relative_to(ROOT) else str(tile)
            stage["threads"] = threads
        elif stage["type"] == "writers.gdal":
            stage["filename"] = str(out_path.relative_to(ROOT))
            if mode in ("chm", "chm_normalized"):
                stage.update(
                    {
                        "origin_x": grid["min_x"],
                        "origin_y": grid["min_y"],
                        "width": grid["n"],
                        "height": grid["m"],
                    }
                )
        elif "filename" in stage and stage["type"].startswith("writers."):
            stage["filename"] = str(out_path.relative_to(ROOT))

    written = RESULTS / "pipelines" / f"pdal_{mode}.json"
    written.parent.mkdir(parents=True, exist_ok=True)
    written.write_text(json.dumps(pipeline, indent=2) + "\n")
    return written


def command_for(tool, mode, executable, out_path, timing_out, threads, grid, tile: Path = TILE):
    """The command line that runs `mode` with `tool`."""
    if tool == "pyfor":
        return [
            executable, str(BENCH / "pyfor_ops.py"), mode,
            "--tile", str(tile), "--out", str(out_path), "--timing-out", str(timing_out),
        ]
    if tool == "lidR":
        return [
            executable, str(BENCH / "lidr.R"), mode,
            "--tile", str(tile), "--out", str(out_path), "--timing-out", str(timing_out),
            "--grid-xmin", repr(grid["min_x"]), "--grid-ymin", repr(grid["min_y"]),
            "--grid-ncols", str(grid["n"]), "--grid-nrows", str(grid["m"]),
            "--grid-res", repr(grid["res"]),
        ]
    if tool == "pdal":
        return [
            executable, "pipeline",
            str(effective_pipeline(mode, threads, out_path, grid, tile)),
        ]
    raise ValueError(tool)


def measure(command, runs: int, warmup: int = 1, cwd: Path = ROOT):
    """Wall clock of `command` in a fresh process, warmup runs discarded."""
    timings = []
    for i in range(warmup + runs):
        start = time.perf_counter()
        proc = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
        elapsed = time.perf_counter() - start
        if proc.returncode != 0:
            raise RuntimeError(
                "command failed:\n  {}\n{}".format(" ".join(command), proc.stderr[-2000:])
            )
        if i >= warmup:
            timings.append(elapsed)

    timings.sort()
    return {
        "runs": runs,
        "median": timings[len(timings) // 2],
        "min": timings[0],
        "max": timings[-1],
        "all": timings,
    }


def op_seconds(timing_out: Path):
    """The in-process operation time a tool reported, or None."""
    if not timing_out.exists():
        return None
    try:
        return json.loads(timing_out.read_text())["op_seconds"]
    except (KeyError, ValueError):
        return None


def run_benchmarks(tools, modes, runs, threads):
    RESULTS.mkdir(parents=True, exist_ok=True)
    grid = pyfor_grid()
    timings = {}
    outputs = {}
    skipped = []

    for mode in modes:
        for tool, executable in tools.items():
            if mode not in TOOL_MODES[tool]:
                skipped.append((tool, mode))
                continue

            extension = OUTPUT_EXTENSIONS.get(mode, ".out")
            out_path = RESULTS / f"{tool}_{mode}{extension}"
            timing_out = RESULTS / f"{tool}_{mode}.timing.json"
            for stale in (out_path, timing_out):
                if stale.exists():
                    stale.unlink()

            command = command_for(tool, mode, executable, out_path, timing_out, threads, grid)
            print(f"  {tool:6} {mode:16}", end="", flush=True)
            try:
                measured = measure(command, runs)
            except RuntimeError as error:
                print(f"FAILED: {str(error).splitlines()[0]}")
                timings[(tool, mode)] = {"error": str(error)}
                continue

            measured["op"] = op_seconds(timing_out)
            measured["command"] = " ".join(command)
            timings[(tool, mode)] = measured
            if mode in ("chm", "chm_normalized"):
                outputs[mode, tool] = out_path
            print(f"{measured['median']:8.3f} s (median of {runs})")

    return timings, outputs, skipped


def raster_agreement(left: Path, right: Path):
    """Compares two rasters cell by cell, honoring nodata and grid offsets.

    The grids of the three tools line up on an integer number of cells or they do not line up at all.
    When they do, each raster is cropped to the overlap and compared. When they do not, the sub pixel
    offset is reported instead, because the cells are simply not the same cells.
    """
    import numpy as np
    import rasterio

    with rasterio.open(left) as a, rasterio.open(right) as b:
        array_a = a.read(1, masked=True).filled(np.nan)
        array_b = b.read(1, masked=True).filled(np.nan)

        offset_x = (b.transform.c - a.transform.c) / a.transform.a
        # The pixel height is negative for a north up raster, so use its magnitude: the offset is
        # counted in pixels downwards from the top, like a row index.
        offset_y = (b.transform.f - a.transform.f) / abs(a.transform.e)
        result = {
            "shape_a": list(array_a.shape),
            "shape_b": list(array_b.shape),
            "pixel_offset": [float(offset_x), float(offset_y)],
        }

        if abs(offset_x - round(offset_x)) > 1e-6 or abs(offset_y - round(offset_y)) > 1e-6:
            result["same_shape"] = array_a.shape == array_b.shape
            result["grids_aligned"] = False
            result["note"] = (
                "grid origins differ by a fraction of a pixel, the cells are not the same cells"
            )
            if result["same_shape"]:
                # What comparing the two rasters by index gives, which is what happens if the grids
                # are assumed to line up.
                both = np.isfinite(array_a) & np.isfinite(array_b)
                difference = np.abs(array_a[both] - array_b[both])
                result["naive_mean_abs_diff"] = (
                    float(difference.mean()) if difference.size else None
                )
                result["naive_max_abs_diff"] = (
                    float(difference.max()) if difference.size else None
                )
            return result

        # Crop both rasters to the cells they have in common.
        start_b_x, start_b_y = int(round(offset_x)), int(round(offset_y))
        start_a_x, start_a_y = max(0, -start_b_x), max(0, -start_b_y)
        start_b_x, start_b_y = max(0, start_b_x), max(0, start_b_y)
        width = min(array_a.shape[1] - start_a_x, array_b.shape[1] - start_b_x)
        height = min(array_a.shape[0] - start_a_y, array_b.shape[0] - start_b_y)

        window_a = array_a[
            start_a_y : start_a_y + height, start_a_x : start_a_x + width
        ]
        window_b = array_b[
            start_b_y : start_b_y + height, start_b_x : start_b_x + width
        ]
        both = np.isfinite(window_a) & np.isfinite(window_b)
        difference = np.abs(window_a[both] - window_b[both])

        result.update(
            {
                "same_shape": True,
                "grids_aligned": True,
                "shape": [height, width],
                "cells": int(window_a.size),
                "cells_compared": int(both.sum()),
                "cells_only_a": int((np.isfinite(window_a) & ~np.isfinite(window_b)).sum()),
                "cells_only_b": int((~np.isfinite(window_a) & np.isfinite(window_b)).sum()),
                "max_abs_diff": float(difference.max()) if difference.size else None,
                "mean_abs_diff": float(difference.mean()) if difference.size else None,
                "identical": bool(difference.size and difference.max() < 1e-6),
            }
        )
        return result


def grid_convention_effect(tile: Path = TILE, cell_size: float = 1.0):
    """What snapping the grid origin and following GDAL's tie rule change, on a tile.

    Computes the maximum z per cell twice, once on the grid pyfor uses now (origin snapped to a
    multiple of the cell size, a point on a grid line belonging to the cell above it) and once on the
    grid it used before (origin at the extent of the data, a point on a grid line belonging to the
    cell below it), then compares the two cell by cell. Both grids are the same shape on the shipped
    tile, which is what makes them comparable as if they were the same cells, and is exactly the
    comparison a user would make by accident.
    """
    import laspy
    import numpy as np

    import pyfor

    las = laspy.read(tile)
    x, y, z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)

    def maxima(origin_x, origin_y, n, m, snap_rows):
        bins_x = np.clip(
            np.floor((x - origin_x) / cell_size).astype(np.int64), 0, n - 1
        )
        if snap_rows:
            bins_y = np.clip(
                m - 1 - np.floor((y - origin_y) / cell_size).astype(np.int64), 0, m - 1
            )
        else:
            bins_y = np.clip(
                np.floor((origin_y + m * cell_size - y) / cell_size).astype(np.int64),
                0,
                m - 1,
            )

        flat = bins_y * n + bins_x
        top = np.full(n * m, -np.inf)
        np.maximum.at(top, flat, z)
        counts = np.bincount(flat, minlength=n * m)
        return np.where(counts > 0, top, np.nan).reshape(m, n)

    spec = pyfor.rasterizer.GridSpec.covering(x.min(), y.min(), x.max(), y.max(), cell_size)
    snapped = maxima(
        spec.origin_x, spec.origin_y, spec.n, spec.m, snap_rows=True
    )

    # The grid pyfor produced before: left edge at the minimum x, top edge at the maximum y.
    old_n = int(np.ceil((x.max() - x.min()) / cell_size))
    old_m = int(np.ceil((y.max() - y.min()) / cell_size))
    old_origin_y = float(y.max()) - old_m * cell_size
    anchored = maxima(x.min(), old_origin_y, old_n, old_m, snap_rows=False)

    both = np.isfinite(snapped) & np.isfinite(anchored)
    difference = np.abs(snapped[both] - anchored[both])
    return {
        "snapped_origin": [spec.origin_x, spec.origin_y],
        "previous_origin": [float(x.min()), old_origin_y],
        "cells_in_common": int(both.sum()),
        "cells_with_a_different_maximum": int((difference > 1e-6).sum()),
        "mean_abs_diff": float(difference.mean()),
        "max_abs_diff": float(difference.max()),
    }


def aligned_tile(path: Path):
    """Writes a synthetic tile whose extent sits on the cell lattice.

    On such a tile pyfor's convention (anchor the grid at the data extent) and the convention lidR
    and PDAL use (snap the grid to the resolution) describe the same cells, so a cell by cell
    comparison of the three tools is exact.

    Two details make the comparison exact rather than approximately exact:

    * Interior points sit in the middle of a cell, so no point lies on an interior cell boundary. A
      point exactly on a boundary is a tie, and the tools break ties differently (see
      ``boundary_tie_probe``).
    * The extent is defined by four points with a height below the local surface, and every cell on
      the outer ring of the grid also gets a point in its middle. The tools disagree about the fate
      of a point sitting exactly on the outer edge (pyfor clips it into the outermost cell, lidR
      grows the raster, PDAL drops it), so those points are kept out of every cell maximum.
    """
    import laspy
    import numpy as np

    rng = np.random.default_rng(20260909)
    size = 50

    # Interior points, one per randomly drawn cell position, in the middle of the cell.
    i = rng.integers(1, size - 1, 20000)
    j = rng.integers(1, size - 1, 20000)

    # One point in the middle of every cell on the outer ring, so no outer cell depends on a point
    # that sits on the edge of the extent.
    ring = np.arange(size)
    ring_x = np.concatenate([ring, ring, np.full(size, 0), np.full(size, size - 1)])
    ring_y = np.concatenate([np.full(size, 0), np.full(size, size - 1), ring, ring])
    i = np.concatenate([i, ring_x])
    j = np.concatenate([j, ring_y])

    x = 405000.0 + i + 0.5
    y = 3276300.0 + j + 0.5
    # A deterministic height surface with many equal values, plus a unique spike inside one cell.
    z = 30.0 + (i % 7) / 2 + (j % 5) / 4
    x = np.concatenate([x, [405012.5]])
    y = np.concatenate([y, [3276337.5]])
    z = np.concatenate([z, [45.0]])

    # The extent is carried by four points on the outer edge, below the surface everywhere.
    x = np.concatenate([x, [405000.0, 405050.0, 405000.0, 405050.0]])
    y = np.concatenate([y, [3276300.0, 3276300.0, 3276350.0, 3276350.0]])
    z = np.concatenate([z, [0.0, 0.0, 0.0, 0.0]])

    header = laspy.LasHeader(point_format=1, version="1.3")
    header.offsets = [400000.0, 3200000.0, 0.0]
    header.scales = [0.01, 0.01, 0.01]
    las = laspy.LasData(header)
    las.x = x
    las.y = y
    las.z = z
    las.intensity = rng.integers(0, 1000, len(x)).astype(np.uint16)
    las.return_number = np.ones(len(x), dtype=np.uint8)
    las.write(path)
    return path


def boundary_tie_probe(tools, threads):
    """Records which cell each tool assigns to a point that sits exactly on a horizontal grid line.

    Two cells, one column, y from 0 to 2. One point at y = 1.0 is exactly on the line between them
    and carries a distinctive height, so the row its height appears in says which cell it landed in.
    """
    import laspy
    import numpy as np
    import rasterio

    path = RESULTS / "tile_boundary.laz"
    header = laspy.LasHeader(point_format=1, version="1.3")
    header.offsets = [0.0, 0.0, 0.0]
    header.scales = [0.01, 0.01, 0.01]
    las = laspy.LasData(header)
    las.x = np.array([0.50, 0.60, 0.55])
    las.y = np.array([0.50, 2.00, 1.00])
    las.z = np.array([1.0, 2.0, 99.0])
    las.write(path)

    grid = {"min_x": 0.0, "min_y": 0.0, "max_x": 1.0, "max_y": 2.0, "n": 1, "m": 2, "res": 1.0}
    probe = {}
    for tool, executable in tools.items():
        out_path = RESULTS / f"{tool}_chm_boundary.tif"
        command = command_for(
            tool, "chm", executable, out_path, RESULTS / "unused.json", threads, grid, path
        )
        subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
        with rasterio.open(out_path) as source:
            column = source.read(1, masked=True).filled(np.nan)[:, 0]
        probe[tool] = [
            None if np.isnan(value) else float(value) for value in column
        ]

    probe["reading"] = (
        "rows are north to south; the point at y = 1.0 on the grid line has z = 99, the point "
        "below it has z = 1 and the point above it has z = 2"
    )
    return probe


def verify_grid_conventions(tools, threads):
    """Runs the CHM operation on the aligned tile and compares the three rasters exactly."""
    tile = aligned_tile(RESULTS / "tile_aligned.laz")
    grid = pyfor_grid(tile)
    rasters = {}

    for tool, executable in tools.items():
        out_path = RESULTS / f"{tool}_chm_aligned.tif"
        command = command_for(
            tool, "chm", executable, out_path, RESULTS / "unused.json", threads, grid, tile
        )
        subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
        rasters[tool] = out_path

    if "pyfor" not in rasters:
        return {}

    verification = {}
    for tool, path in rasters.items():
        if tool != "pyfor":
            verification[tool] = raster_agreement(rasters["pyfor"], path)
    return verification


def verify_outputs(outputs):
    """Cross checks the rasters the tools produced, so the timing table is not comparing noise."""
    verification = {}
    for mode in ("chm", "chm_normalized"):
        produced = {tool: path for (m, tool), path in outputs.items() if m == mode}
        if "pyfor" not in produced:
            continue
        for tool, path in produced.items():
            if tool == "pyfor":
                continue
            if not path.exists() or path.stat().st_size == 0:
                verification[f"{mode}:{tool}"] = {"error": "no raster was written"}
                continue
            verification[f"{mode}:{tool}"] = raster_agreement(produced["pyfor"], path)
    return verification


def count_metric_layers():
    """How many metrics each tool wrote, since the two suites do not contain the same metrics.

    pyfor's metrics operation returns rasters in memory and writes nothing, so it has no entry here.
    """
    import rasterio

    counts = {}
    for path in sorted(RESULTS.glob("*_metrics.tif")):
        tool = path.name.split("_", 1)[0]
        with rasterio.open(path) as source:
            # The band count is the number of metrics the tool computed and wrote.
            counts[tool] = source.count
    return counts


def environment(pdal, rscript, threads):
    import numpy
    import laspy
    import pyfor

    info = {
        "cpu": next(
            (line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo") if "model name" in line),
            platform.processor(),
        ),
        "cores": os.cpu_count(),
        "memory_gb": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "pyfor": pyfor.__version__,
        "laspy": laspy.__version__,
        "numpy": numpy.__version__,
        "threads_per_tool": threads,
        "tile": str(TILE.relative_to(ROOT)),
        "tile_bytes": TILE.stat().st_size,
    }
    if rscript:
        probe = subprocess.run(
            [rscript, "-e", 'cat(R.version.string, "| lidR ", as.character(packageVersion("lidR")), '
                            '" | rlas ", as.character(packageVersion("rlas")), sep = "")'],
            capture_output=True, text=True,
        )
        info["r"] = probe.stdout.strip() or probe.stderr.strip()
    if pdal:
        probe = subprocess.run([pdal, "--version"], capture_output=True, text=True)
        info["pdal"] = " ".join(probe.stdout.split()[:4])
    return info


def markdown(timings, tools, modes, skipped=()):
    lines = [
        "| operation | tool | median (s) | min (s) | max (s) | in-process op (s) |",
        "|---|---|---|---|---|---|",
    ]
    skipped = set(skipped)
    for mode in modes:
        for tool in tools:
            measured = timings.get((tool, mode))
            if not measured:
                label = "not supported" if (tool, mode) in skipped else "not run"
                lines.append(f"| {mode} | {tool} | {label} | | | |")
                continue
            if "error" in measured:
                lines.append(f"| {mode} | {tool} | failed | | | |")
                continue
            op = measured["op"]
            lines.append(
                "| {} | {} | {:.3f} | {:.3f} | {:.3f} | {} |".format(
                    mode, tool, measured["median"], measured["min"], measured["max"],
                    "-" if op is None else f"{op:.3f}",
                )
            )
    return "\n".join(lines)


def describe_agreement(key, value):
    """One line description of a raster comparison."""
    if "error" in value:
        return f"{key}: {value['error']}"
    if not value.get("grids_aligned", value.get("same_shape", False)):
        described = (
            f"{key}: grids differ (pixel offset {value['pixel_offset'][0]:.4f}, "
            f"{value['pixel_offset'][1]:.4f}), {value.get('note', '')}"
        )
        if "naive_max_abs_diff" in value:
            described += (
                f"; compared by index anyway: mean |diff| {value['naive_mean_abs_diff']:.3g}, "
                f"max |diff| {value['naive_max_abs_diff']:.3g}"
            )
        return described
    return (
        f"{key}: {value['cells_compared']}/{value['cells']} cells comparable "
        f"(pyfor only {value['cells_only_a']}, other only {value['cells_only_b']}), "
        f"max |diff| {value['max_abs_diff']:.3g}, mean |diff| {value['mean_abs_diff']:.3g}, "
        f"identical: {value['identical']}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3, help="timed runs per measurement")
    parser.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    parser.add_argument("--tools", nargs="+", default=TOOLS, choices=TOOLS)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--pdal", default=shutil.which("pdal") or os.environ.get("PDAL"),
                        help="path to the pdal binary")
    parser.add_argument("--rscript", default=shutil.which("Rscript"))
    parser.add_argument("--no-verify-aligned", dest="verify_aligned", action="store_false",
                        help="skip the aligned tile raster comparison")
    args = parser.parse_args()

    found = available_tools(args.pdal, args.rscript)
    requested = {name: exe for name, exe in found.items() if name in args.tools}
    missing = sorted(set(args.tools) - set(requested))
    if missing:
        print(f"unavailable on this machine, skipped: {', '.join(missing)}", file=sys.stderr)
    if not requested:
        raise SystemExit("no requested tool is available")

    print(f"tools: {', '.join(requested)}")
    print(f"tile:  {TILE.relative_to(ROOT)} ({TILE.stat().st_size / 1024:.0f} KiB)")
    print(f"runs:  {args.runs} per measurement, {args.threads} thread per tool\n")

    timings, outputs, skipped = run_benchmarks(requested, args.modes, args.runs, args.threads)
    verification = verify_outputs(outputs)
    aligned = verify_grid_conventions(requested, args.threads) if args.verify_aligned else {}
    ties = boundary_tie_probe(requested, args.threads) if args.verify_aligned else {}

    report = {
        "environment": environment(args.pdal, args.rscript, args.threads),
        "tools": {name: exe for name, exe in requested.items()},
        "modes": args.modes,
        "skipped": [f"{tool}:{mode}" for tool, mode in skipped],
        "timings": {
            f"{tool}:{mode}": measured for (tool, mode), measured in timings.items()
        },
        "verification": verification,
        "verification_aligned_tile": aligned,
        "boundary_ties": ties,
        "metrics_layers": count_metric_layers(),
        "grid_convention_effect": grid_convention_effect(),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "results.json").write_text(json.dumps(report, indent=2) + "\n")

    print("\n" + markdown(timings, list(requested), args.modes, skipped))
    if verification:
        print("\noutput agreement against pyfor, each tool on the tile as distributed:")
        for key, value in verification.items():
            print("  " + describe_agreement(key, value))
    if aligned:
        print("\noutput agreement against pyfor, on a synthetic tile whose extent sits on the cell")
        print("lattice and whose interior points avoid cell boundaries (identical grids everywhere):")
        for key, value in aligned.items():
            print("  " + describe_agreement(f"chm:{key}", value))
    if ties:
        print("\na point exactly on a horizontal cell boundary (z = 99 at y = 1.0, rows north to south):")
        for tool in requested:
            if tool in ties:
                print(f"  {tool:6} {ties[tool]}")
    print(f"\nwrote {RESULTS / 'results.json'}")


if __name__ == "__main__":
    main()
