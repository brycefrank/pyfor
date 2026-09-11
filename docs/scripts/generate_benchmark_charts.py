#!/usr/bin/env python3
"""Generate the benchmark charts for the docs from a benchmark run.

Reads ``benchmarks/results/results.json`` (written by ``benchmarks/bench.py``)
and writes two SVG charts into ``docs/src/assets/``:

* ``bench-wallclock.svg``  wall clock per operation, all three tools
* ``bench-inprocess.svg``  in-process cost per operation, pyfor and lidR

PDAL reports only wall clock, so it has no in-process series. The charts are
committed and inlined into the page, so they can be regenerated only where a
results file exists; run the benchmark first if it is missing.

The SVG uses Starlight's CSS custom properties and ``currentColor``, so the
charts follow the site's light and dark themes.

Usage: python docs/scripts/generate_benchmark_charts.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "benchmarks" / "results" / "results.json"
OUT_DIR = REPO_ROOT / "docs" / "src" / "assets"

TOOLS = ["pyfor", "lidR", "pdal"]
MODES = ["read", "readwrite", "normalize", "chm", "chm_normalized", "metrics"]
MODE_LABELS = {
    "read": "read",
    "readwrite": "read + write",
    "normalize": "normalize",
    "chm": "chm",
    "chm_normalized": "normalized chm",
    "metrics": "metrics",
}

# Series colours. pyfor is the accent; the others are the text colour at low
# opacity, which reads correctly on both themes.
SERIES_FILL = {
    "pyfor": "var(--sl-color-accent-high)",
    "lidR": "currentColor",
    "pdal": "currentColor",
}
SERIES_OPACITY = {"pyfor": "1", "lidR": "0.42", "pdal": "0.2"}

WIDTH = 720
GUTTER = 132
PLOT_LEFT = 140
PLOT_RIGHT = WIDTH - 46
PLOT_WIDTH = PLOT_RIGHT - PLOT_LEFT


def nice_axis(value: float) -> tuple[float, float]:
    """A rounded axis maximum and tick step that cover `value`."""
    for step in (0.2, 0.25, 0.5, 1.0, 2.0, 2.5, 5.0, 10.0, 20.0):
        if value / step <= 4.5:
            ticks = int(value / step) + 1
            return step * ticks, step
    step = 5.0
    return step * (int(value / step) + 1), step


def bars_for(chart: str, timings: dict) -> dict[str, list[float | None]]:
    """Values per tool for every operation, in seconds."""
    key = "median" if chart == "wallclock" else "op"
    series: dict[str, list[float | None]] = {}
    for tool in TOOLS:
        values = []
        for mode in MODES:
            entry = timings.get(f"{tool}:{mode}")
            values.append(None if entry is None else entry.get(key))
        series[tool] = values
    return series


def render_chart(chart: str, series: dict[str, list[float | None]], unit: str) -> str:
    tools = [tool for tool in TOOLS if any(value is not None for value in series[tool])]
    tallest = max(value for tool in tools for value in series[tool] if value is not None)
    axis_max, step = nice_axis(tallest)

    bar_height = 12
    bar_gap = 3
    group_gap = 14
    legend_y = 14
    ruler_y = 40
    plot_top = 48
    group_height = len(tools) * bar_height + (len(tools) - 1) * bar_gap
    height = int(plot_top + len(MODES) * group_height + (len(MODES) - 1) * group_gap + 12)

    def x_of(value: float) -> float:
        return PLOT_LEFT + (value / axis_max) * PLOT_WIDTH

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {height}" '
        f'role="img" aria-label="{unit} per operation for {", ".join(tools)}" '
        f'font-family="inherit" font-size="12">'
    ]

    # Legend.
    legend_x = PLOT_LEFT
    for tool in tools:
        parts.append(
            f'<rect x="{legend_x}" y="{legend_y - 9}" width="11" height="11" rx="2" '
            f'fill="{SERIES_FILL[tool]}" fill-opacity="{SERIES_OPACITY[tool]}"/>'
        )
        parts.append(
            f'<text x="{legend_x + 17}" y="{legend_y}" fill="currentColor" font-size="12.5">'
            f"{tool}</text>"
        )
        legend_x += 17 + 8 * len(tool) + 22

    # Ruler and gridlines.
    ticks = []
    tick = 0.0
    while tick <= axis_max + 1e-9:
        ticks.append(round(tick, 4))
        tick += step
    for tick in ticks:
        x = x_of(tick)
        parts.append(
            f'<line x1="{x:.1f}" y1="{ruler_y}" x2="{x:.1f}" y2="{height - 10}" '
            f'stroke="currentColor" stroke-opacity="0.12"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{ruler_y - 7}" fill="currentColor" fill-opacity="0.65" '
            f'font-size="10.5" text-anchor="middle">{tick:g}</text>'
        )
    parts.append(
        f'<text x="{PLOT_RIGHT}" y="{legend_y}" fill="currentColor" fill-opacity="0.65" '
        f'font-size="11" text-anchor="end">{unit}</text>'
    )
    parts.append(
        f'<line x1="{PLOT_LEFT}" y1="{ruler_y}" x2="{PLOT_RIGHT}" y2="{ruler_y}" '
        f'stroke="currentColor" stroke-opacity="0.35"/>'
    )

    # Bars, grouped by operation.
    for index, mode in enumerate(MODES):
        top = plot_top + index * (group_height + group_gap)
        label_y = top + group_height / 2 + 4
        parts.append(
            f'<text x="{GUTTER}" y="{label_y:.1f}" fill="currentColor" font-size="13" '
            f'text-anchor="end">{MODE_LABELS[mode]}</text>'
        )
        for offset, tool in enumerate(tools):
            value = series[tool][index]
            if value is None:
                continue
            y = top + offset * (bar_height + bar_gap)
            width = max(x_of(value) - PLOT_LEFT, 1.5)
            parts.append(
                f'<rect x="{PLOT_LEFT}" y="{y:.1f}" width="{width:.1f}" height="{bar_height}" '
                f'rx="2" fill="{SERIES_FILL[tool]}" fill-opacity="{SERIES_OPACITY[tool]}"/>'
            )
            parts.append(
                f'<text x="{PLOT_LEFT + width + 6:.1f}" y="{y + bar_height - 2:.1f}" '
                f'fill="currentColor" fill-opacity="0.8" font-size="11.5">{value:.2f}</text>'
            )

    parts.append("</svg>")
    return "\n".join(parts)


def main() -> int:
    if not RESULTS.is_file():
        print(f"error: {RESULTS.relative_to(REPO_ROOT)} not found; run benchmarks/bench.py first")
        return 1

    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    timings = results["timings"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    charts = {
        "bench-wallclock.svg": render_chart("wallclock", bars_for("wallclock", timings), "seconds"),
        "bench-inprocess.svg": render_chart("inprocess", bars_for("inprocess", timings), "seconds"),
    }
    for name, svg in charts.items():
        (OUT_DIR / name).write_text(svg + "\n", encoding="utf-8")
        print(f"wrote {(OUT_DIR / name).relative_to(REPO_ROOT)} ({len(svg)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
