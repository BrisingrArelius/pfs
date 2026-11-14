"""Shared, easily edited throttled-SSD experiment settings.

Change FAST_TO_SLOW_RATIO to select an experiment. The output directory
follows the ratio automatically. Rates are per-target MiB/s.
"""

FAST_MIB_S = 45
FAST_TO_SLOW_RATIO = 10.0
RESULTS_SUBDIR = f"throttled_ssd_{float(FAST_TO_SLOW_RATIO)}"

if FAST_MIB_S <= 0 or FAST_TO_SLOW_RATIO < 1:
    raise ValueError("Expected a positive fast rate and a ratio >= 1")
SLOW_MIB_S = FAST_MIB_S / FAST_TO_SLOW_RATIO
FAST_BYTES_S = round(FAST_MIB_S * 1048576)
SLOW_BYTES_S = round(SLOW_MIB_S * 1048576)
if SLOW_BYTES_S < 1:
    raise ValueError("Slow rate must be at least one byte/s")
if RESULTS_SUBDIR in ("", ".", "..") or "/" in RESULTS_SUBDIR or "\\" in RESULTS_SUBDIR:
    raise ValueError("RESULTS_SUBDIR must be one directory name")
