"""pyfor side of the cross tool benchmark.

Each mode runs one comparable operation on a single tile and writes its output to the path given by
``--out`` where the operation produces one. Times are measured by ``bench.py`` around this process,
so nothing here should print transient information.

Usage: python pyfor_ops.py <mode> --tile <tile> --out <path>
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyfor import cloud  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=["read", "readwrite", "normalize", "chm", "chm_normalized", "metrics"],
    )
    parser.add_argument("--tile", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--timing-out", help="where to write the in-process operation time")
    args = parser.parse_args()

    # Only the operation is timed, so the interpreter and library import time is not counted here.
    # bench.py measures that separately, as the wall clock of the whole process.
    start = time.perf_counter()
    pc = cloud.Cloud(args.tile)

    if args.mode == "read":
        # Touch the data so the read cannot be optimized away.
        _ = float(pc.data.points["z"].sum())

    elif args.mode == "readwrite":
        pc.write(args.out)

    elif args.mode == "normalize":
        pc.normalize(1)
        _ = float(pc.data.points["z"].sum())

    elif args.mode == "chm":
        pc.grid(1).raster("max", "z").write(args.out)

    elif args.mode == "chm_normalized":
        pc.normalize(1)
        pc.grid(1).raster("max", "z").write(args.out)

    elif args.mode == "metrics":
        pc.normalize(1)
        pc.grid(20).standard_metrics(2)

    if args.timing_out:
        with open(args.timing_out, "w") as handle:
            json.dump({"op_seconds": time.perf_counter() - start}, handle)


if __name__ == "__main__":
    main()
