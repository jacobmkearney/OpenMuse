import numpy as np
import pandas as pd
import os
import datetime as dt
from typing import List, Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt

# --- Prior knowledge ---

### Constructor Information

# Muse S Athena specs (From the [Muse website](https://eu.choosemuse.com/products/muse-s-athena) - note that these info might not be up to date or fully accurate):
# - Wireless Connection: BLE 5.3, 2.4 GHz
# - EEG Channels: 4 EEG channels (TP9, AF7, AF8, TP10) + 1 (or 4?) amplified Aux channels
#   - Sample Rate: 256 Hz
#   - Sample Resolution: 14 bits / sample
# - Accelerometer: Three-axis at 52Hz, 16-bit resolution, range +/- 2G
# - Gyroscope: Three-axis at 52Hz, 16-bit resolution, range +/- 250dps
# - PPG Sensor: Triple wavelength: IR (850nm), Near-IR (730nm), Red (660nm), 64 Hz sample rate, 20-bit resolution
# - fNIRS Sensor: 5-optode bilateral frontal cortex hemodynamics, 64 Hz sample rate, 20-bit resolution
#   - Might result in 1, 4, 5, 8, 16 OPTICS channels

### Presets

# Different presets enable/disable some channels, but the exact combinations are not fully documented.
# - p20-p61: Red LED in the centre is off
# - p1034, p1041, p1042, p1043: red LED in the centre is brightly on (suggesting the activation of OPTICS or PPG channels)
# - p1035, p1044, p1045, p4129: red LED in the centre is dimmer

# Based on these specs, we can derive the following plausible expectations regarding each channel type:


KNOWN_RATES = {
    "EEG": 256.0,
    "AUX": 256.0,
    "ACC": 52.0,
    "GYRO": 52.0,
    "PPG": 64.0,
    "OPTICS": 64.0,
}

KNOWN_CHANNELS = {
    "EEG": [0, 4],  #  256 Hz, 14 bits
    "AUX": [0, 1, 4],  # 256 Hz, 14 bits
    "ACC": [0, 3],  #  52 Hz, 16 bits
    "GYRO": [0, 3],  #  52 Hz, 16 bits
    "PPG": [0, 3],  # 64 Hz, 20 bits
    "OPTICS": [0, 1, 4, 5, 8, 16],  # 64 Hz, 20 bits
}

# Each data file (./data_raw/) should contain some combination of these channels.
# The exact combination depends on the preset used during recording.
# Importantly, these channels types are likely indistinguishable from the data alone, so it is best to group them according to their data characteristics in GROUPS, namely CH256, CH52, CH64.

# --- Expectations ---


EXPECTED_GROUPS = {
    "CH256": list(
        set(
            KNOWN_CHANNELS["EEG"]
            + KNOWN_CHANNELS["AUX"]
            + [
                i + j
                for i in KNOWN_CHANNELS["EEG"]
                for j in KNOWN_CHANNELS["AUX"]
                if j > 0
            ]
        )
    ),
    "CH52": list(
        set(
            KNOWN_CHANNELS["ACC"]
            + KNOWN_CHANNELS["GYRO"]
            + [
                i + j
                for i in KNOWN_CHANNELS["ACC"]
                for j in KNOWN_CHANNELS["GYRO"]
                if j > 0
            ]
        )
    ),
    "CH64": list(
        set(
            KNOWN_CHANNELS["PPG"]
            + KNOWN_CHANNELS["OPTICS"]
            + [
                i + j
                for i in KNOWN_CHANNELS["PPG"]
                for j in KNOWN_CHANNELS["OPTICS"]
                if j > 0
            ]
        )
    ),
}

# Expected sampling rates (Hz) and bits per sample for each candidate channel type
EXPECTED_RATES = {
    "CH256": 256.0,
    "CH52": 52.0,
    "CH64": 64.0,
}
BITS_PER_SAMPLE = {
    "CH256": 14,  # treat bit-width as a constraint, not packing
    "CH52": 16,
    "CH64": 20,
}


# ======================================================================
# Functions ============================================================
# ======================================================================
def parse_lines(lines):
    """
    Each line has three tab-separated fields:
    - Timestamp (ISO 8601 with microseconds and timezone)
    - UUID / session ID
    - Hex payload (the actual Muse packet)
    """
    fromiso = dt.datetime.fromisoformat
    tobytes = bytes.fromhex

    times, uuids, data = [], [], []
    for line in lines:
        parts = line.strip().split("\t")
        if len(parts) != 3:
            print(f"Skipping malformed line: {line.strip()}")
            continue
        times.append(fromiso(parts[0].replace("Z", "+00:00")).timestamp())
        uuids.append(parts[1])
        data.append(tobytes(parts[2]))

    # If more than one UUID, warn
    if len(set(uuids)) > 1:
        print(f"Warning: Multiple UUIDs found: {set(uuids)}")

    return times, uuids, data


# Decode ACC+GYR (52Hz) packets ==================================================
def _decode_ch52_method1(raw_bytes: bytes) -> np.ndarray:
    """
    Decode a candidate CH52 raw packet to an array of shape (n_samples, 6)
    with dtype=int16 and little-endian ordering. Returns an empty (0,6)
    array when decoding fails or the packet is not CH52-like.

    Based on the patterns of the brute-force method observed in analyze_rawdata.py
    """
    # Known CH52 packet headers (kept as a set for fast membership test)
    CH52_HEADERS = {
        0xEA,
        0xED,
        0xDF,
        0xE6,
        0xCB,
        0xD3,
        0xDE,
        0xDB,
    }

    if len(raw_bytes) < 1:
        return np.empty((0, 6), dtype=np.int16)

    header = raw_bytes[0]
    if header not in CH52_HEADERS:
        return np.empty((0, 6), dtype=np.int16)

    payload = raw_bytes[1:]
    # Each sample has 6 channels × 2 bytes = 12 bytes
    bytes_per_sample = 12
    n = len(payload) // bytes_per_sample
    if n == 0:
        return np.empty((0, 6), dtype=np.int16)

    usable = payload[: n * bytes_per_sample]
    # little-endian int16
    data = np.frombuffer(usable, dtype="<i2")
    try:
        samples = data.reshape(n, 6)
    except Exception:
        # In case of unexpected shape, return empty consistent type
        return np.empty((0, 6), dtype=np.int16)
    return samples


def _sign_extend_12(v: int) -> int:
    return v - 0x1000 if (v & 0x800) else v


def unpack_12bit_le(data: bytes) -> List[int]:
    """Unpack little-endian 12-bit signed ints from bytes; returns python ints."""
    out: List[int] = []
    n3 = (len(data) // 3) * 3
    i = 0
    while i < n3:
        b0 = data[i]
        b1 = data[i + 1]
        b2 = data[i + 2]
        s0 = (b0 | ((b1 & 0x0F) << 8)) & 0xFFF
        s1 = (((b1 >> 4) | (b2 << 4))) & 0xFFF
        out.append(_sign_extend_12(s0))
        out.append(_sign_extend_12(s1))
        i += 3
    return out


def decode_12bit_rows_from_region(sample_region: bytes) -> Optional[np.ndarray]:
    """Trim to 3-byte groups, unpack, trim to multiple of 6, return ndarray (N,6) int16 or None."""
    trimmed = sample_region[: (len(sample_region) // 3) * 3]
    if len(trimmed) == 0:
        return None
    ints12 = unpack_12bit_le(trimmed)
    mvals = (len(ints12) // 6) * 6
    if mvals == 0:
        return None
    rows = np.array(ints12[:mvals], dtype=np.int16).reshape(-1, 6)
    return rows


def _decode_ch52_method2(
    raw_bytes: bytes, pid_offset: int = 9, post_pid_skip: int = 4
) -> np.ndarray:
    """
    Header-agnostic scan for packet-id with (high_nibble==4 and low_nibble==7).
    pid_offset: preferred fixed pid index to check first (repo uses 9).
    post_pid_skip: number of bytes to skip after pid before sample region.
    Returns ndarray shape (n_samples, 6) dtype=int16 or empty array.

    Based on https://github.com/AbosaSzakal/MuseAthenaDataformatParser
    """
    if len(raw_bytes) < max(10, pid_offset + 1):
        return np.empty((0, 6), dtype=np.int16)

    matches = []
    # Preferred fixed check
    try:
        pid = raw_bytes[pid_offset]
        if (pid >> 4) == 4 and (pid & 0x0F) == 7:
            matches.append(pid_offset)
    except Exception:
        pass

    # Robust scan (but avoid too many false positives by skipping obvious header areas?)
    for idx, b in enumerate(raw_bytes):
        if (b >> 4) == 4 and (b & 0x0F) == 7:
            matches.append(idx)

    # deduplicate (preserve order)
    seen = set()
    matches_unique = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            matches_unique.append(m)
    if not matches_unique:
        return np.empty((0, 6), dtype=np.int16)

    all_samples = []
    for pid_idx in matches_unique:
        sample_start = pid_idx + 1 + post_pid_skip
        if sample_start >= len(raw_bytes):
            continue
        sample_region = raw_bytes[sample_start:]
        # Trim region to multiple of 3 bytes
        sample_region = sample_region[: (len(sample_region) // 3) * 3]
        if len(sample_region) == 0:
            continue
        ints12 = unpack_12bit_le(sample_region)
        # Trim to multiple of 6 values (6 values per sample row)
        mvals = (len(ints12) // 6) * 6
        if mvals == 0:
            continue
        ints12 = ints12[:mvals]
        arr = np.array(ints12, dtype=np.int16).reshape(-1, 6)
        all_samples.append(arr)

    if not all_samples:
        return np.empty((0, 6), dtype=np.int16)
    return np.vstack(all_samples)


# Run decoding ============================================================
def decode_channels(lines: List[str]) -> pd.DataFrame:
    """
    Decode lines (from parse_lines) and return a DataFrame with columns:
    Time, ACCx, ACCy, ACCz, GYRx, GYRy, GYRz
    Time is unix timestamp (float seconds). Timestamps for multi-sample
    packets are distributed evenly at CH52 rate, ending at the packet time.
    """
    times, uuids, data = parse_lines(lines)

    all_ch52 = []
    all_times = []

    rate = EXPECTED_RATES["CH52"]

    for t_packet, raw in zip(times, data):
        ch52 = _decode_ch52_method2(raw)
        if ch52.size == 0:
            continue

        n = ch52.shape[0]
        if n == 1:
            ts = [t_packet]
        else:
            # Place samples evenly in time so that the last sample aligns with t_packet.
            # Sample interval = 1 / rate. First sample time = t_packet - (n-1)/rate
            dt_interval = 1.0 / rate
            start_t = t_packet - (n - 1) * dt_interval
            ts = list(start_t + np.arange(n) * dt_interval)

        all_ch52.append(ch52)
        all_times.extend(ts)

    if not all_ch52:
        cols = ["Time", "GYRx", "GYRy", "GYRz", "ACCx", "ACCy", "ACCz"]
        return pd.DataFrame(columns=cols)

    stacked = np.vstack(all_ch52)
    df = pd.DataFrame(stacked, columns=["GYRx", "GYRy", "GYRz", "ACCx", "ACCy", "ACCz"])
    df.insert(0, "Time", all_times)
    return df


# ======================================================================
# Debugging ============================================================
# ======================================================================
def inspect_packet_simple(
    raw_bytes: bytes,
    pid_offset: int = 9,
    post_pid_skip: int = 4,
    scan_all: bool = True,
    max_candidates: int = 6,
    stats: bool = True,
    verbose: bool = False,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Decode candidate IMU regions and return (stacked_rows or None, meta).
    meta contains minimal programmatic info; verbose prints short human-readable lines.
    """
    meta: Dict[str, Any] = {
        "pid_offset": pid_offset,
        "found_pids": [],
        "candidates": [],
        "per_candidate": [],
        "rows_count": 0,
        "notes": [],
    }

    # find pid matches
    found = [i for i, b in enumerate(raw_bytes) if (b >> 4) == 4 and (b & 0x0F) == 7]
    meta["found_pids"] = found

    # build candidate list: prefer pid_offset if it matches; optionally scan_all
    candidates: List[int] = []
    if (
        len(raw_bytes) > pid_offset
        and (raw_bytes[pid_offset] >> 4) == 4
        and ((raw_bytes[pid_offset] & 0x0F) == 7)
    ):
        candidates.append(pid_offset)
    if scan_all:
        for i in found:
            if i not in candidates:
                candidates.append(i)
    if not candidates and len(raw_bytes) > pid_offset:
        candidates.append(pid_offset)
        meta["notes"].append("fallback_to_pid_offset")
    meta["candidates"] = candidates[:max_candidates]

    all_rows = []
    for idx in meta["candidates"]:
        cmeta: Dict[str, Any] = {"idx": int(idx)}
        if idx >= len(raw_bytes):
            cmeta["note"] = "idx_out_of_range"
            meta["per_candidate"].append(cmeta)
            continue
        sample_start = idx + 1 + post_pid_skip
        cmeta["sample_start"] = int(sample_start)
        if sample_start >= len(raw_bytes):
            cmeta["note"] = "start_beyond"
            meta["per_candidate"].append(cmeta)
            continue
        region = raw_bytes[sample_start:]
        rows = decode_12bit_rows_from_region(region)
        if rows is None:
            cmeta["note"] = "no_rows"
            meta["per_candidate"].append(cmeta)
            continue
        cmeta.update(
            {"rows": int(rows.shape[0]), "trim_len": int((len(region) // 3) * 3)}
        )
        meta["per_candidate"].append(cmeta)
        all_rows.append(rows)
        if verbose:
            print(f"cand idx={idx} start={sample_start} rows={rows.shape[0]}")
    if not all_rows:
        meta["notes"].append("no_decoded_rows")
        return None, meta

    stacked = np.vstack(all_rows)
    meta["rows_count"] = int(stacked.shape[0])

    if stats:
        gyr = stacked[:, 0:3].astype(float)
        acc = stacked[:, 3:6].astype(float)
        acc_mag = np.sqrt((acc**2).sum(axis=1))
        meta["stats"] = {
            "gyr_mean": list(np.mean(gyr, axis=0)),
            "gyr_std": list(np.std(gyr, axis=0)),
            "acc_mean": list(np.mean(acc, axis=0)),
            "acc_std": list(np.std(acc, axis=0)),
            "acc_mag_mean": float(acc_mag.mean()),
            "acc_mag_std": float(acc_mag.std()),
        }
        if verbose:
            print("stats:", meta["stats"])

    return stacked, meta


# --- Robust scoring (compact) ---------------------------------------------
def score_result_simple(rows: Optional[np.ndarray]) -> Dict[str, float]:
    """Simple, bounded scoring to pick best candidate; returns score and key diagnostics."""
    if rows is None or not isinstance(rows, np.ndarray) or rows.size == 0:
        return {
            "score": 0.0,
            "rows": 0.0,
            "accel_mean_abs": 0.0,
            "gyro_mean_abs": 0.0,
            "acc_mag_std": 0.0,
        }

    gyr = rows[:, 0:3].astype(float)
    acc = rows[:, 3:6].astype(float)
    N = rows.shape[0]

    rows_norm = min(N, 30) / 30.0  # cap contribution
    accel_mean_abs = float(np.mean(np.abs(np.mean(acc, axis=0))))
    gyro_mean_abs = float(np.mean(np.abs(np.mean(gyr, axis=0))))
    acc_mag_std = float(np.std(np.sqrt((acc**2).sum(axis=1))))

    # simple normalized qualities (0..1). expected accel magnitude unknown; we only compare relative
    accel_q = np.tanh(accel_mean_abs / 1000.0)  # grows to 1 for large DC offsets
    gyro_q = 1.0 - np.tanh(gyro_mean_abs / 1000.0)  # near-zero gyro -> close to 1
    stability_q = 1.0 - np.tanh(acc_mag_std / 200.0)  # small std -> close to 1

    # weights chosen to favor decoded blocks with clear accel DC and low gyro bias
    score = 1.5 * rows_norm + 1.0 * accel_q + 1.0 * gyro_q + 0.5 * stability_q
    return {
        "score": float(score),
        "rows": float(N),
        "accel_mean_abs": accel_mean_abs,
        "gyro_mean_abs": gyro_mean_abs,
        "acc_mag_std": acc_mag_std,
    }


# --- Compact grid-runner and summary --------------------------------------
def debug_file(
    lines: List[str],
    n_packets_per_file: int = 3,
    pid_offsets: List[int] = [8, 9, 10],
    post_pid_skips: List[int] = [3, 4, 5],
    top_k: int = 3,
    verbose_inspect: bool = False,
) -> Dict[int, Dict[str, Any]]:
    """
    Run a small grid per-packet and return summary_per_packet:
      {packet_index: {"timestamp":..., "best": {...}, "top": [...], "all": [...]}}
    Also prints short per-packet lines and a per-file aggregate.
    """
    times, uuids, raws = parse_lines(lines)
    packets_to_check = min(len(raws), n_packets_per_file)
    summary_per_packet: Dict[int, Dict[str, Any]] = {}
    best_counts: Dict[Tuple[int, int], int] = {}

    for i in range(packets_to_check):
        raw = raws[i]
        grid_entries = []
        for po in pid_offsets:
            for ps in post_pid_skips:
                rows, meta = inspect_packet_simple(
                    raw,
                    pid_offset=po,
                    post_pid_skip=ps,
                    scan_all=False,
                    stats=False,
                    verbose=verbose_inspect,
                )
                metrics = score_result_simple(rows)
                grid_entries.append(
                    {
                        "pid": int(po),
                        "pskip": int(ps),
                        "score": metrics["score"],
                        "rows": int(metrics["rows"]),
                        "accel_mean_abs": metrics["accel_mean_abs"],
                        "gyro_mean_abs": metrics["gyro_mean_abs"],
                        "acc_mag_std": metrics["acc_mag_std"],
                    }
                )

        # sort and pick top candidates
        grid_entries.sort(key=lambda e: e["score"], reverse=True)
        best = grid_entries[0]
        top = grid_entries[:top_k]
        key = (best["pid"], best["pskip"])
        best_counts[key] = best_counts.get(key, 0) + 1

        summary_per_packet[i] = {
            "timestamp": times[i],
            "best": best,
            "top": top,
            "all": grid_entries,
        }

        # concise print
        print(
            f"Pkt {i} @ {times[i]}  BEST pid={best['pid']} pskip={best['pskip']} score={best['score']:.2f} rows={best['rows']} accel={best['accel_mean_abs']:.1f}"
        )

    # per-file aggregate
    if best_counts:
        sorted_counts = sorted(best_counts.items(), key=lambda kv: kv[1], reverse=True)
        (pid_m, ps_m), cnt = sorted_counts[0]
        stability = cnt / max(1, packets_to_check)
        print(
            f"\nMost common best: pid={pid_m} pskip={ps_m} (count={cnt}/{packets_to_check}) stability={stability:.2f}"
        )
        for (pid_, pskip_), c in sorted_counts:
            print(f"  ({pid_},{pskip_}) -> {c}")
    return summary_per_packet


# ======================================================================
# Main =================================================================
# ======================================================================
if __name__ == "__main__":
    all_results = []
    for fname in os.listdir("data_raw"):
        # fname = "data_p1034.txt"
        with open(os.path.join("data_raw", fname), "r", encoding="utf-8") as f:
            lines = f.readlines()
        print(f"\n=== file: {fname} ({len(lines)} lines) ===")
        debug_file(lines)
        df = decode_channels(lines)
        # df.plot(
        #     x="Time", y=["ACCx", "ACCy", "ACCz", "GYRx", "GYRy", "GYRz"], subplots=True
        # )
        df["Preset"] = fname.replace("data_", "").replace(".txt", "")
        all_results.append(df)

    combined = pd.concat(all_results, ignore_index=True)

    # Visualize results
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(15, 10))
    for (preset, group), ax in zip(combined.groupby("Preset"), axes.flatten()):
        group.plot(
            x="Time",
            y=["ACCx", "ACCy", "ACCz", "GYRx", "GYRy", "GYRz"],
            title=preset,
            ax=ax,
            alpha=0.7,
        )
