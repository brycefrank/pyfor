# lidR side of the cross tool benchmark.
#
# Usage: Rscript lidr.R <mode> --tile <tile> --out <path> [--timing-out <path>] [--threads n]
#
# Modes mirror the pyfor and PDAL sides: read, readwrite, normalize, chm, chm_normalized, metrics.
# lidR runs single threaded here: no future plan is set, so parallelization is left to the default
# of one worker. The metrics operation includes lidR's normalizer, as it does on the pyfor side.

suppressPackageStartupMessages({
  library(lidR)
  library(terra)
})

args <- commandArgs(trailingOnly = TRUE)
mode <- args[1]
get_arg <- function(flag, default = NULL) {
  i <- match(flag, args)
  if (is.na(i)) default else args[i + 1]
}
tile <- get_arg("--tile")
out <- get_arg("--out")
timing_out <- get_arg("--timing-out")

# The raster grid to produce, so that every tool is compared on the same cells. Defaults to a grid
# anchored at the extent of the data, which is what pyfor does.
grid_xmin <- as.numeric(get_arg("--grid-xmin", NA))
grid_ymin <- as.numeric(get_arg("--grid-ymin", NA))
grid_ncols <- as.integer(get_arg("--grid-ncols", NA))
grid_nrows <- as.integer(get_arg("--grid-nrows", NA))
grid_res <- as.numeric(get_arg("--grid-res", 1))
template <- NULL
if (!is.na(grid_xmin) && !is.na(grid_ncols)) {
  template <- rast(
    ncols = grid_ncols, nrows = grid_nrows,
    xmin = grid_xmin, xmax = grid_xmin + grid_ncols * grid_res,
    ymin = grid_ymin, ymax = grid_ymin + grid_nrows * grid_res
  )
}

# The operation is timed as a whole, which includes reading the tile, so that it matches what the
# other tools measure. Package loading happens before the clock starts.
start <- proc.time()[["elapsed"]]

las <- readLAS(tile, filter = "")

if (mode == "read") {
  invisible(sum(las$Z))
} else if (mode == "readwrite") {
  writeLAS(las, out)
} else if (mode == "normalize") {
  normalized <- normalize_height(las, tin())
  invisible(sum(normalized$Z))
} else if (mode == "chm") {
  chm <- rasterize_canopy(las, 1, p2r(), template = template)
  writeRaster(chm, out, overwrite = TRUE)
} else if (mode == "chm_normalized") {
  normalized <- normalize_height(las, tin())
  chm <- rasterize_canopy(normalized, 1, p2r(), template = template)
  writeRaster(chm, out, overwrite = TRUE)
} else if (mode == "metrics") {
  normalized <- normalize_height(las, tin())
  metrics <- pixel_metrics(
    normalized,
    ~ stdmetrics(X, Y, Z, Intensity, ReturnNumber, Classification),
    res = 20
  )
  writeRaster(metrics, out, overwrite = TRUE)
} else {
  stop("unknown mode: ", mode)
}

if (!is.null(timing_out)) {
  elapsed <- proc.time()[["elapsed"]] - start
  cat(sprintf('{"op_seconds": %.6f}\n', elapsed), file = timing_out)
}
