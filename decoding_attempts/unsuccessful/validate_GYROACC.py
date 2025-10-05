import numpy as np
import pandas as pd
import datetime as dt
import struct
import os
import binascii
import collections
from collections import Counter, defaultdict
import random


def _parse_lines(lines):
    fromiso = dt.datetime.fromisoformat
    tobytes = bytes.fromhex
    times, uuids, data = [], [], []
    for line in lines:
        parts = line.strip().split("\t")
        times.append(fromiso(parts[0].replace("Z", "+00:00")).timestamp())
        uuids.append(parts[1])
        data.append(tobytes(parts[2]))
    return times, uuids, data


def parse_acc_gyro(lines: list[str], debug=True) -> pd.DataFrame:
    """
    Parse Muse ACC/GYRO packets.
    - Packets tagged with 0x47.
    - After tag byte, skip 4 metadata bytes.
    - Then read 18 int16 little-endian values = 3 samples * 6 channels.
    - ACC scaling: ±2g -> 1 g = 16384 counts
    - GYRO scaling: ±250 dps -> 1 dps ≈ 131 counts
    - Timestamps spaced at 52 Hz, aligned so payload time = last sample.
    """
    times, uuids, data = _parse_lines(lines)

    all_values, all_sample_times = [], []
    sample_rate = 52.0
    dt_sample = 1.0 / sample_rate

    dumped = False

    for payload_time, payload in zip(times, data):
        pos = 0
        plen = len(payload)
        while pos < plen:
            tag = payload[pos]
            # if not target tag, skip forward
            if tag != 0x47:
                pos += 1
                continue

            # ensure enough bytes for tag + 4 metadata + 36 data
            min_needed = pos + 1 + 4 + 36
            if min_needed > plen:
                # advance past tag to avoid stalling and continue scanning
                pos += 1
                continue

            payload_start = pos + 1 + 4
            end_index = payload_start + 36
            block = payload[payload_start:end_index]
            try:
                vals = list(struct.unpack("<18h", block))
            except struct.error:
                pos += 1
                continue

            values = np.array(vals, dtype=np.int16).reshape((3, 6))
            if debug and not dumped:
                print("First ACC/GYRO packet dump (18 raw int16):", vals)
                dumped = True

            acc = values[:, 0:3].astype(np.float32) / 16384.0
            gyro = values[:, 3:6].astype(np.float32) / 131.0
            scaled_values = np.hstack((acc, gyro))

            this_times = np.array(
                [payload_time - (2 - i) * dt_sample for i in range(3)], dtype=np.float64
            )

            all_values.append(scaled_values)
            all_sample_times.append(this_times)

            pos = end_index

    if not all_values:
        print("No valid ACC/GYRO data parsed.")
        return pd.DataFrame()

    all_values = np.vstack(all_values)
    all_sample_times = np.concatenate(all_sample_times)

    df = pd.DataFrame(
        all_values, columns=["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    )
    df["time"] = all_sample_times

    # sanity check
    observed_dur = all_sample_times[-1] - all_sample_times[0]
    expected_dur = (len(all_sample_times) - 1) * dt_sample
    print(
        f"Observed duration: {observed_dur:.2f} s, expected {expected_dur:.2f} s from {len(all_sample_times)} samples"
    )

    return df


if __name__ == "__main__":
    data_dir = "./data_raw/"
    files = sorted(os.listdir(data_dir))

    all_dfs = {}

    for filename in files:
        print(f"Processing {filename}...")
        with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as f:
            lines = f.readlines()
        df = parse_acc_gyro(lines, debug=False)

        # Compute Observed Frequency
        time_diffs = np.diff(df["time"].values)
        observed_freq = 1.0 / np.median(time_diffs)
        print(f"Observed sampling frequency: {observed_freq:.2f} Hz")

        all_dfs[filename] = df

    all_dfs["data_p50.txt"].plot(
        x="time",
        y=["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
        subplots=True,
    )


# Goals:
# - Confirm that GYRO/ACC data should be 16-bits per sample
# - Verify scaling/endian/ordering: Ensure that "<18h" matches the device spec. A wrong column offset will mix bytes from a different sensor and create spikes.
# - Ensure you are parsing only ACC blocks: Search the payload for other tags; an accidental offset that lands you inside another block type will yield nonsense values. Confirm each 0x47 block contains 18 int16 repeatedly.


import numpy as np
import pandas as pd
import datetime as dt
import struct
import os
from typing import Tuple, List, Dict, Any


# helper from your original script
def _parse_lines(lines: List[str]):
    fromiso = dt.datetime.fromisoformat
    tobytes = bytes.fromhex
    times, uuids, data = [], [], []
    for line in lines:
        parts = line.strip().split("\t")
        times.append(fromiso(parts[0].replace("Z", "+00:00")).timestamp())
        uuids.append(parts[1])
        data.append(tobytes(parts[2]))
    return times, uuids, data


def _validate_47_block(
    payload: bytes,
    pos: int,
    payload_time: float,
    *,
    sample_rate: float = 52.0,
    acc_scale: float = 16384.0,
    gyro_scale: float = 131.0,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Validate a candidate 0x47 block at payload[pos] and return a diagnostic dict.
    Expects a minimum layout: [0x47][4 metadata bytes][N bytes...]
    We require at least one 36-byte ACC/GYRO record starting at pos+1+4.
    """
    report: Dict[str, Any] = {
        "pos": pos,
        "has_valid_record": False,
        "num_records": 0,
        "unpack_errors": 0,
        "out_of_range_counts": 0,
        "out_of_range_scaled": 0,
        "nearby_tag_conflicts": [],
        "examples": [],
    }

    plen = len(payload)
    tag = payload[pos]
    if tag != 0x47:
        return report

    payload_start = pos + 1 + 4  # tag + 4 metadata
    if payload_start + 36 > plen:
        report["unpack_errors"] = 1
        return report

    # Search for other 0x47 bytes inside the window that would indicate a different block boundary
    search_window_end = min(
        plen, payload_start + 36 * 8
    )  # allow checking multiple consecutive records
    for i in range(payload_start, search_window_end):
        if payload[i] == 0x47 and i != pos:
            # store relative offset(s) of conflicting tag bytes
            report["nearby_tag_conflicts"].append(i - pos)

    # Try to unpack successive 36-byte records until we run out of space or unpack fails
    cur = payload_start
    records = []
    while cur + 36 <= plen:
        block = payload[cur : cur + 36]
        try:
            vals = struct.unpack("<18h", block)
        except struct.error:
            report["unpack_errors"] += 1
            break

        arr = np.array(vals, dtype=np.int16).reshape((3, 6))
        records.append(arr)
        report["num_records"] += 1
        cur += 36

    if not records:
        return report

    # Flatten records for checks
    all_vals = np.vstack(records)  # shape (n_samples, 6)
    acc_counts = all_vals[:, 0:3].astype(np.float32)
    gyro_counts = all_vals[:, 3:6].astype(np.float32)

    # scaled values
    acc_g = acc_counts / acc_scale
    gyro_dps = gyro_counts / gyro_scale

    # plausibility checks
    # generous absolute cutoffs to detect obviously wrong endian or column-swapped data
    acc_abs_limit_g = 8.0  # if ACC outside ±8 g something is certainly wrong
    gyro_abs_limit_dps = (
        2000.0  # if GYRO outside ±2000 dps something is certainly wrong
    )

    out_acc = np.logical_or(np.abs(acc_g) > acc_abs_limit_g, np.isnan(acc_g))
    out_gyro = np.logical_or(np.abs(gyro_dps) > gyro_abs_limit_dps, np.isnan(gyro_dps))

    report["out_of_range_counts"] = int(
        np.count_nonzero(out_acc) + np.count_nonzero(out_gyro)
    )

    # check relative spikes that indicate misaligned columns byte-swapped mixes etc.
    # compute sample-to-sample diffs and flag if many diffs are huge
    diffs = np.abs(np.diff(np.vstack((acc_g, gyro_dps)).T, axis=0))
    spike_threshold = 4.0  # change-in-g or change-in-dps threshold (g and dps mixed scale; thresholds could be per-axis)
    # count diffs exceeding threshold on any channel
    spike_counts = int(np.count_nonzero(np.any(diffs > spike_threshold, axis=1)))
    report["spike_counts"] = spike_counts
    report["num_samples"] = all_vals.shape[0]

    # fraction-of-samples outside expected device nominal ranges (±3g for acc, ±500 dps for gyro)
    nom_acc_limit = 3.0
    nom_gyro_limit = 500.0
    frac_bad_acc = np.count_nonzero(np.abs(acc_g) > nom_acc_limit) / (acc_g.size)
    frac_bad_gyro = np.count_nonzero(np.abs(gyro_dps) > nom_gyro_limit) / (
        gyro_dps.size
    )
    report["frac_bad_acc"] = frac_bad_acc
    report["frac_bad_gyro"] = frac_bad_gyro

    # save a couple of illustrative samples (first up to 4)
    n_examples = min(4, all_vals.shape[0])
    for i in range(n_examples):
        s_acc = tuple(float(x) for x in (acc_g[i, 0], acc_g[i, 1], acc_g[i, 2]))
        s_gyro = tuple(
            float(x) for x in (gyro_dps[i, 0], gyro_dps[i, 1], gyro_dps[i, 2])
        )
        report["examples"].append({"acc_g": s_acc, "gyro_dps": s_gyro})

    # if most samples look plausible mark as valid
    if (
        report["num_samples"] > 0
        and frac_bad_acc < 0.2
        and frac_bad_gyro < 0.2
        and report["out_of_range_counts"] == 0
    ):
        report["has_valid_record"] = True

    if verbose:
        print("Validation report for tag at pos", pos, ":", report)

    return report


def parse_acc_gyro_with_validation(
    lines: List[str], debug: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Parse ACC/GYRO as before but validate each found 0x47 block and produce a diagnostics report.
    Returns (df, diagnostics)
    """
    times, uuids, data = _parse_lines(lines)

    all_values, all_sample_times = [], []
    validations = []
    sample_rate = 52.0
    dt_sample = 1.0 / sample_rate

    for payload_time, payload in zip(times, data):
        pos = 0
        plen = len(payload)
        while pos < plen:
            tag = payload[pos]
            if tag != 0x47:
                pos += 1
                continue

            # quick bounds check
            payload_start = pos + 1 + 4
            end_index = payload_start + 36
            if end_index > plen:
                # not enough room for even one record; record and skip
                validations.append(
                    {
                        "time": payload_time,
                        "pos": pos,
                        "error": "short_block",
                        "bytes_left": plen - pos,
                    }
                )
                pos += 1
                continue

            # Call the validator which will attempt to unpack repeated 18h records
            report = _validate_47_block(payload, pos, payload_time, verbose=False)
            report["time"] = payload_time
            validations.append(report)

            # If validator says we have valid records, ingest them the same way as original code:
            if report["has_valid_record"]:
                scaled_list = []
                cur = payload_start
                while cur + 36 <= plen:
                    block = payload[cur : cur + 36]
                    try:
                        vals = struct.unpack("<18h", block)
                    except struct.error:
                        break
                    values = np.array(vals, dtype=np.int16).reshape((3, 6))
                    acc = values[:, 0:3].astype(np.float32) / 16384.0
                    gyro = values[:, 3:6].astype(np.float32) / 131.0
                    scaled = np.hstack((acc, gyro))
                    scaled_list.append(scaled)
                    cur += 36

                if scaled_list:
                    all_scaled = np.vstack(scaled_list)  # shape (num_samples, 6)
                    num_samples = all_scaled.shape[0]
                    # Assume samples are in chronological order (oldest first); space uniformly backward from payload_time
                    time_start = payload_time - (num_samples - 1) * dt_sample
                    this_times = np.array(
                        [time_start + i * dt_sample for i in range(num_samples)],
                        dtype=np.float64,
                    )
                    all_values.append(all_scaled)
                    all_sample_times.append(this_times)

                pos = cur
            else:
                # suspicious block: advance one byte to continue scanning; the validator captured diagnostics
                pos += 1

    diagnostics = {
        "total_blocks_found": sum(1 for v in validations if v.get("pos") is not None),
        "valid_blocks": sum(1 for v in validations if v.get("has_valid_record")),
        "short_blocks": sum(1 for v in validations if v.get("error") == "short_block"),
        "validation_details": validations,
    }

    if not all_values:
        return pd.DataFrame(), diagnostics

    all_values = np.vstack(all_values)
    all_sample_times = np.concatenate(all_sample_times)

    df = pd.DataFrame(
        all_values, columns=["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    )
    df["time"] = all_sample_times

    # quick sanity print
    observed_dur = all_sample_times[-1] - all_sample_times[0]
    expected_dur = (len(all_sample_times) - 1) * dt_sample
    if debug:
        print(
            f"Observed duration: {observed_dur:.2f} s, expected {expected_dur:.2f} s from {len(all_sample_times)} samples"
        )

    return df, diagnostics


def make_diagnostics_summary(
    all_diags: Dict[str, Dict[str, Any]],
    all_dfs: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build a one-row-per-file summary DataFrame from diagnostics and parsed dfs.
    Returns the summary DataFrame (and optionally writes CSV).
    """
    rows = []
    for fname, diag in all_diags.items():
        df = all_dfs.get(fname, pd.DataFrame())
        n_samples = 0 if df.empty else len(df)
        median_freq = np.nan
        if n_samples > 1:
            diffs = np.diff(df["time"].values)
            median_freq = 1.0 / np.median(diffs)

        # aggregate validation metrics
        details = diag.get("validation_details", [])
        total_blocks = diag.get("total_blocks_found", sum(1 for _ in details))
        valid_blocks = diag.get(
            "valid_blocks", sum(1 for d in details if d.get("has_valid_record"))
        )
        short_blocks = diag.get(
            "short_blocks", sum(1 for d in details if d.get("error") == "short_block")
        )
        # compute aggregates of frac_bad_acc / frac_bad_gyro where present
        frac_bad_accs = [
            d.get("frac_bad_acc") for d in details if d.get("frac_bad_acc") is not None
        ]
        frac_bad_gyros = [
            d.get("frac_bad_gyro")
            for d in details
            if d.get("frac_bad_gyro") is not None
        ]
        mean_frac_bad_acc = float(np.mean(frac_bad_accs)) if frac_bad_accs else np.nan
        mean_frac_bad_gyro = (
            float(np.mean(frac_bad_gyros)) if frac_bad_gyros else np.nan
        )

        # count blocks with nearby tag conflicts or unpack errors
        conflicts = sum(1 for d in details if d.get("nearby_tag_conflicts"))
        unpack_errors = sum(1 for d in details if d.get("unpack_errors", 0) > 0)

        rows.append(
            {
                "filename": fname,
                "n_samples": int(n_samples),
                "median_freq": (
                    float(median_freq) if not np.isnan(median_freq) else np.nan
                ),
                "total_blocks": int(total_blocks),
                "valid_blocks": int(valid_blocks),
                "short_blocks": int(short_blocks),
                "blocks_with_conflicts": int(conflicts),
                "blocks_with_unpack_errors": int(unpack_errors),
                "mean_frac_bad_acc": mean_frac_bad_acc,
                "mean_frac_bad_gyro": mean_frac_bad_gyro,
            }
        )

    summary_df = pd.DataFrame(rows).set_index("filename").sort_index()

    # Add some high-level derived columns
    summary_df["pct_valid_blocks"] = (
        summary_df["valid_blocks"] / summary_df["total_blocks"]
    ).replace([np.inf, -np.inf], np.nan)
    summary_df["has_data"] = summary_df["n_samples"] > 0

    return summary_df


# Replace your __main__ loop with this pattern to collect diagnostics and print a global summary:
if __name__ == "__main__":
    data_dir = "./data_raw/"
    files = sorted(os.listdir(data_dir))

    all_dfs = {}
    all_diags = {}

    for filename in files:
        print(f"Processing {filename}...")
        path = os.path.join(data_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        df, diag = parse_acc_gyro_with_validation(lines, debug=False)
        all_dfs[filename] = df
        all_diags[filename] = diag

        # brief per-file summary
        print(
            f"  samples: {len(df) if not df.empty else 0}, "
            f"valid_blocks: {diag.get('valid_blocks',0)}, "
            f"total_blocks: {diag.get('total_blocks_found',0)}, "
            f"short_blocks: {diag.get('short_blocks',0)}"
        )

    # Build and display the aggregated summary
    summary = make_diagnostics_summary(all_diags, all_dfs)
    pd.set_option("display.width", 200)
    pd.set_option(
        "display.float_format", lambda x: f"{x:.3f}" if not np.isnan(x) else "NaN"
    )
    print("\n=== Global diagnostics summary ===")
    print(summary.to_markdown(floatfmt=(".2f")))


# INSPECT FILES ----------------------------------------------------------------


def inspect_file(
    path: str,
    parse_fn,  # function: lines -> (df, diagnostics)
    payload_map_threshold: int = 10000,
    show_top_conflicts: int = 5,
) -> Dict[str, Any]:
    """
    Inspect a single raw input file and produce summaries and forensic helpers.
    parse_fn must be a callable that accepts file lines and returns (df, diagnostics)
    as implemented earlier (parse_acc_gyro_with_validation).

    Returns a dict with:
      - df: parsed DataFrame
      - diagnostics: raw diagnostics from parse_fn
      - summary_df: one-row summary (pandas.DataFrame)
      - conflict_offset_counter: Counter of conflict offsets
      - conflict_examples: list of selected diagnostic entries with hexdumps and interpreted int16s
    """

    # --- local helpers (defined inside inspect_file) ---
    def _read_lines(p: str) -> List[str]:
        with open(p, "r", encoding="utf-8") as f:
            return f.readlines()

    def _build_payload_map(lines: List[str]) -> List[Tuple[float, bytes]]:
        """Return list of (time, payload_bytes). Kept in file order for lookups."""
        from datetime import datetime as _dt

        tobytes = bytes.fromhex
        pairs = []
        for line in lines:
            parts = line.strip().split("\t")
            t = _dt.fromisoformat(parts[0].replace("Z", "+00:00")).timestamp()
            pairs.append((t, tobytes(parts[2])))
        return pairs

    def _conflict_offset_hist(diag: Dict[str, Any]) -> collections.Counter:
        offsets = []
        for d in diag.get("validation_details", []):
            for off in d.get("nearby_tag_conflicts", []):
                offsets.append(off)
        return collections.Counter(offsets)

    def _print_offset_histogram(counter: collections.Counter, top: int = 30) -> None:
        items = sorted(counter.items(), key=lambda kv: kv[1], reverse=True)[:top]
        if not items:
            print("No nearby-tag conflicts recorded.")
            return
        print("Top conflict offsets (offset: count):")
        for off, cnt in items:
            print(f"  {off:4d}: {cnt:6d}")

    def _hexdump_region(
        payload: bytes, pos: int, before: int = 32, after: int = 128
    ) -> str:
        a = max(0, pos - before)
        b = min(len(payload), pos + after)
        chunk = payload[a:b]
        hexs = binascii.hexlify(chunk).decode()
        grouped = " ".join(hexs[i : i + 2] for i in range(0, len(hexs), 2))
        header = f"hex [{a}:{b}] (pos {pos} shown at byte index {pos-a}):"
        return f"{header}\n{grouped}"

    def _interpret_int16_lines(chunk: bytes, little: bool = True, cols: int = 6) -> str:
        fmt = "<h" if little else ">h"
        n = (len(chunk) // 2) * 2
        vals = [struct.unpack(fmt, chunk[i : i + 2])[0] for i in range(0, n, 2)]
        rows = [vals[i : i + cols] for i in range(0, len(vals), cols)]
        lines = []
        for r in rows:
            lines.append(", ".join(f"{v:6d}" for v in r))
        return "\n".join(lines) if lines else "<no int16 data>"

    def _select_conflict_entries(diag: Dict[str, Any], n: int) -> List[Dict[str, Any]]:
        entries = [
            d
            for d in diag.get("validation_details", [])
            if d.get("nearby_tag_conflicts")
        ]
        entries = sorted(
            entries, key=lambda d: len(d.get("nearby_tag_conflicts")), reverse=True
        )
        return entries[:n]

    # --- main logic ---
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    print(f"Inspecting file: {path}")
    lines = _read_lines(path)

    # Build payload map (time -> payload). Only keep if file not huge.
    payload_pairs = (
        _build_payload_map(lines) if len(lines) <= payload_map_threshold else []
    )
    if payload_pairs:
        print(f"Built payload map for {len(payload_pairs)} payload lines.")
    else:
        print("Skipping payload map (file too large to hold full payload map).")

    # Parse using provided parse function
    df, diagnostics = parse_fn(lines)

    # Build one-row summary DataFrame
    n_samples = 0 if df.empty else len(df)
    median_freq = np.nan
    if n_samples > 1:
        diffs = np.diff(df["time"].values)
        median_freq = float(1.0 / np.median(diffs))

    details = diagnostics.get("validation_details", [])
    total_blocks = diagnostics.get("total_blocks_found", sum(1 for _ in details))
    valid_blocks = diagnostics.get(
        "valid_blocks", sum(1 for d in details if d.get("has_valid_record"))
    )
    short_blocks = diagnostics.get(
        "short_blocks", sum(1 for d in details if d.get("error") == "short_block")
    )
    frac_bad_accs = [
        d.get("frac_bad_acc") for d in details if d.get("frac_bad_acc") is not None
    ]
    frac_bad_gyros = [
        d.get("frac_bad_gyro") for d in details if d.get("frac_bad_gyro") is not None
    ]
    mean_frac_bad_acc = float(np.mean(frac_bad_accs)) if frac_bad_accs else np.nan
    mean_frac_bad_gyro = float(np.mean(frac_bad_gyros)) if frac_bad_gyros else np.nan
    conflicts = sum(1 for d in details if d.get("nearby_tag_conflicts"))
    unpack_errors = sum(1 for d in details if d.get("unpack_errors", 0) > 0)

    summary_row = {
        "filename": os.path.basename(path),
        "n_samples": int(n_samples),
        "median_freq": median_freq,
        "total_blocks": int(total_blocks),
        "valid_blocks": int(valid_blocks),
        "short_blocks": int(short_blocks),
        "blocks_with_conflicts": int(conflicts),
        "blocks_with_unpack_errors": int(unpack_errors),
        "mean_frac_bad_acc": mean_frac_bad_acc,
        "mean_frac_bad_gyro": mean_frac_bad_gyro,
    }
    summary_df = pd.DataFrame([summary_row]).set_index("filename")
    summary_df["pct_valid_blocks"] = (
        summary_df["valid_blocks"] / summary_df["total_blocks"]
    ).replace([np.inf, -np.inf], np.nan)
    summary_df["has_data"] = summary_df["n_samples"] > 0

    # Conflict offset histogram
    offset_counter = _conflict_offset_hist(diagnostics)
    print("\nConflict offset histogram (top entries):")
    _print_offset_histogram(offset_counter, top=20)

    # Determine whether conflicts are multiples of 36 (likely adjacent records) or random
    if offset_counter:
        offsets = list(offset_counter.keys())
        multiples = [off for off in offsets if off % 36 == 0]
        if multiples:
            print(
                f"Offsets that are multiples of 36 (example up to 10): {sorted(multiples)[:10]}"
            )
        else:
            print("No conflict offsets that are exact multiples of 36 detected.")

    # Inspect top conflict entries with hexdump and int16 interpretations
    conflict_entries = _select_conflict_entries(diagnostics, show_top_conflicts)
    conflict_examples = []
    if conflict_entries:
        print(
            f"\nInspecting top {len(conflict_entries)} conflict entries (hexdump + int16 interpretation):"
        )
        for e in conflict_entries:
            pos = e.get("pos")
            t = e.get("time")
            num_conf = len(e.get("nearby_tag_conflicts", []))
            rec = {
                "pos": pos,
                "time": t,
                "num_conflicts": num_conf,
                "nearby_tag_conflicts": e.get("nearby_tag_conflicts"),
            }
            # try to find matching payload (by time) if payload map exists
            payload = None
            if payload_pairs:
                # find first exact time match; fallback to nearest
                for tt, pbytes in payload_pairs:
                    if tt == t:
                        payload = pbytes
                        break
                if payload is None:
                    # find nearest by small tolerance
                    times = np.array([tt for tt, _ in payload_pairs])
                    idx = int(np.argmin(np.abs(times - t)))
                    payload = payload_pairs[idx][1]
            # create hexdump and interpreted int16s if payload available
            if payload is not None and pos is not None:
                hexd = _hexdump_region(payload, pos, before=24, after=120)
                payload_start = pos + 1 + 4
                chunk = (
                    payload[payload_start : payload_start + 72]
                    if payload_start < len(payload)
                    else b""
                )
                int16_le = _interpret_int16_lines(chunk, little=True, cols=6)
                int16_be = _interpret_int16_lines(chunk, little=False, cols=6)
                rec.update(
                    {"hexdump": hexd, "int16_le": int16_le, "int16_be": int16_be}
                )
            else:
                rec.update(
                    {
                        "hexdump": "<payload-not-available>",
                        "int16_le": "",
                        "int16_be": "",
                    }
                )
            conflict_examples.append(rec)
            # print a compact preview
            print(f"\n- conflict pos={pos}, time={t}, num_conflicts={num_conf}")
            if payload is not None:
                print(hexd.splitlines()[0])
                print("LE int16 sample rows:")
                print(int16_le.splitlines()[:4] if int16_le else "<none>")
            else:
                print("<payload not available for hexdump/interpretation>")

    else:
        print("\nNo conflict entries to show.")

    # Quick advice hints based on summary
    hints = []
    if summary_df["pct_valid_blocks"].iloc[0] >= 0.5:
        hints.append("High fraction of valid blocks; endian/ordering likely correct.")
    else:
        hints.append(
            "Low valid-block fraction; inspect int16 BE/LE samples for endianness or wrong offset."
        )

    if summary_df["blocks_with_conflicts"].iloc[0] > 0:
        hints.append(
            "Nearby-tag conflicts detected; check whether conflicts fall at 36*n offsets (adjacent records) or in metadata region."
        )

    if summary_df["blocks_with_unpack_errors"].iloc[0] > 0:
        hints.append(
            "Unpack errors found; some blocks don't contain full 36-byte records."
        )

    # Assemble return payload
    result = {
        "df": df,
        "diagnostics": diagnostics,
        "summary_df": summary_df,
        "conflict_offset_counter": offset_counter,
        "conflict_examples": conflict_examples,
        "hints": hints,
    }

    return result


def summarize_inspect_result(inspect_result: dict, max_examples: int = 3) -> dict:
    """
    Produce a compact, human-friendly summary from inspect_file(...) output.

    Returns a dict with:
      - summary_row: the one-row pandas summary (same as result['summary_df'])
      - short_hints: list of short hint strings
      - top_conflict_overview: list of up to max_examples short dicts for conflict_examples
      - counts: key counts (n_conflict_examples, n_unpack_error_blocks, n_total_examples)
    """
    import numpy as _np

    summary_df = inspect_result.get("summary_df")
    hints = inspect_result.get("hints", [])
    conflict_counter = inspect_result.get("conflict_offset_counter", {})
    conflict_examples = inspect_result.get("conflict_examples", [])

    # Basic counts
    n_conflict_examples = len(
        [c for c in conflict_examples if c.get("num_conflicts", 0) > 0]
    )
    n_unpack_errors = sum(
        1
        for d in inspect_result.get("diagnostics", {}).get("validation_details", [])
        if d.get("unpack_errors", 0) > 0
    )
    n_total_examples = len(conflict_examples)

    # Prepare compact examples: show up to max_examples, prefer those with more conflicts or unpack errors
    def _score_example(ex):
        score = 0
        score += ex.get("num_conflicts", 0) * 10
        # unpack_errors info lives in diagnostics; try to surface it if available
        # treat presence of 'int16_le'/'int16_be' as positive for inspection
        if ex.get("int16_le"):
            score += 1
        return score

    sorted_examples = sorted(conflict_examples, key=_score_example, reverse=True)[
        :max_examples
    ]

    short_examples = []
    for ex in sorted_examples:
        # extract a single-line hexdump header and first LE/BE int16 row for compactness
        hexd = ex.get("hexdump", "")
        hexd_header = hexd.splitlines()[0] if hexd else "<no hexdump>"
        int16_le = ex.get("int16_le", "")
        int16_be = ex.get("int16_be", "")
        first_le_row = int16_le.splitlines()[0] if int16_le else ""
        first_be_row = int16_be.splitlines()[0] if int16_be else ""
        short_examples.append(
            {
                "pos": ex.get("pos"),
                "time": ex.get("time"),
                "num_conflicts": ex.get("num_conflicts"),
                "nearby_tag_conflicts": ex.get("nearby_tag_conflicts", [])[
                    :6
                ],  # up to first 6 offsets
                "hexdump_header": hexd_header,
                "first_int16_le": first_le_row,
                "first_int16_be": first_be_row,
            }
        )

    # Compact conflict offset summary: top 5 offsets and counts
    top_offsets = []
    if conflict_counter:
        # conflict_counter may be a Counter or dict
        items = sorted(conflict_counter.items(), key=lambda kv: kv[1], reverse=True)[:5]
        top_offsets = [{"offset": int(k), "count": int(v)} for k, v in items]

    short_hints = []
    # keep hints short and focused
    if summary_df is not None and not summary_df.empty:
        row = summary_df.iloc[0]
        pct_valid = float(row.get("pct_valid_blocks", _np.nan))
        if _np.isnan(pct_valid):
            short_hints.append("pct_valid_blocks: NaN — check diagnostics")
        else:
            short_hints.append(f"pct_valid_blocks: {pct_valid:.2f}")
        short_hints.append(f"n_samples: {int(row.get('n_samples', 0))}")
        short_hints.append(
            f"median_freq: {row.get('median_freq'):.2f}"
            if not _np.isnan(row.get("median_freq", _np.nan))
            else "median_freq: NaN"
        )
        short_hints.append(
            f"blocks_with_conflicts: {int(row.get('blocks_with_conflicts', 0))}"
        )
        short_hints.append(
            f"blocks_with_unpack_errors: {int(row.get('blocks_with_unpack_errors', 0))}"
        )
    else:
        short_hints.append("No summary row available")

    # Merge user-provided hints, but keep unique and short
    for h in hints:
        if len(short_hints) >= 8:
            break
        if h not in short_hints:
            short_hints.append(h if len(h) <= 120 else h[:117] + "...")

    result = {
        "summary_row": summary_df,
        "short_hints": short_hints,
        "top_conflict_overview": short_examples,
        "conflict_offsets_top": top_offsets,
        "counts": {
            "n_conflict_examples": n_conflict_examples,
            "n_unpack_error_blocks": int(n_unpack_errors),
            "n_total_examples": n_total_examples,
        },
    }
    return result


summarize_inspect_result(
    inspect_file(
        "./data_raw/data_p20.txt",
        parse_acc_gyro_with_validation,
        payload_map_threshold=5000,
        show_top_conflicts=3,
    ),
    max_examples=2,
)


def reclassify_conflicts(diagnostics, meta_len=5, record_len=36, top_n=10):
    from collections import Counter

    details = diagnostics.get("validation_details", [])
    counter = Counter()
    reclassified = {"adjacent_ok": 0, "suspicious_inside_record": 0, "other": 0}
    examples = {"adjacent": [], "inside": [], "other": []}

    for d in details:
        offs = d.get("nearby_tag_conflicts", [])
        for off in offs:
            counter[off] += 1
            rel = off - meta_len
            if rel >= 0 and rel % record_len == 0:
                reclassified["adjacent_ok"] += 1
                if len(examples["adjacent"]) < top_n:
                    examples["adjacent"].append((d.get("pos"), off))
            elif 0 <= rel < record_len:
                reclassified["suspicious_inside_record"] += 1
                if len(examples["inside"]) < top_n:
                    examples["inside"].append((d.get("pos"), off))
            else:
                reclassified["other"] += 1
                if len(examples["other"]) < top_n:
                    examples["other"].append((d.get("pos"), off))

    summary = {
        "total_conflicts_reported": sum(counter.values()),
        "counts_by_offset": counter.most_common(10),
        **reclassified,
        "examples": examples,
    }
    return summary


diagnostics = inspect_file(
    "./data_raw/data_p20.txt",
    parse_acc_gyro_with_validation,
    payload_map_threshold=5000,
    show_top_conflicts=3,
)
reclassify_conflicts(diagnostics["diagnostics"])
