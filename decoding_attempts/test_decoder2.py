import os
from typing import List, Tuple, Optional, Dict, Any
import statistics
import numpy as np
import datetime as dt

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

# Goal for this script is to try various decoding approaches and validate
# See: https://github.com/AbosaSzakal/MuseAthenaDataformatParser




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


# ======================================================================
# Debugging ============================================================
# ======================================================================
def inspect_packet(
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
def score_result(rows: Optional[np.ndarray]) -> Dict[str, float]:
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


def build_offset_map(
    data_dir: str = "data_raw",
    files: List[str] = None,
    n_packets_per_file: int = 10,
    pid_offsets: List[int] = [8, 9, 10],
    post_pid_skips: List[int] = [3, 4, 5],
    scan_all: bool = False,
) -> Dict[str, Tuple[int, int]]:
    """
    Grid-run per file and return mapping: filename -> (best_pid_offset, best_post_pid_skip).

    Behavior:
      - If files is None, iterate sorted filenames in data_dir.
      - For each file, parse with parse_lines(lines) -> times, uuids, raws.
      - For first n_packets_per_file packets, run grid (pid_offsets x post_pid_skips).
      - For each packet choose best candidate by score_result_simple(rows).
      - Aggregate best choices across packets; pick the most frequent pair.
      - If tie, choose pair with highest average score across packets where it was best.
    """
    mapping: Dict[str, Tuple[int, int]] = {}
    files_to_check = files if files is not None else sorted(os.listdir(data_dir))

    for fname in files_to_check:
        fpath = os.path.join(data_dir, fname)
        if not os.path.isfile(fpath):
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        times, uuids, raws = parse_lines(lines)
        pkt_count = min(len(raws), n_packets_per_file)
        if pkt_count == 0:
            continue

        # Track how many times each pair was selected and accumulate scores for tie-breaking
        freq: Dict[Tuple[int, int], int] = {}
        scores_for_pair: Dict[Tuple[int, int], List[float]] = {}

        for i in range(pkt_count):
            raw = raws[i]
            # evaluate grid for this packet and collect all candidate scores
            candidates: List[Dict[str, Any]] = []
            for po in pid_offsets:
                for ps in post_pid_skips:
                    rows, meta = inspect_packet(
                        raw,
                        pid_offset=po,
                        post_pid_skip=ps,
                        scan_all=scan_all,
                        stats=False,
                        verbose=False,
                    )
                    metrics = score_result(rows)
                    candidates.append(
                        {
                            "pid": int(po),
                            "pskip": int(ps),
                            "score": float(metrics["score"]),
                        }
                    )
            # pick best candidate for this packet
            candidates.sort(key=lambda e: e["score"], reverse=True)
            best = candidates[0]
            key = (best["pid"], best["pskip"])
            freq[key] = freq.get(key, 0) + 1
            scores_for_pair.setdefault(key, []).append(best["score"])

        # choose most frequent pair; break ties by average score
        if not freq:
            continue
        # sort pairs by (frequency desc, avg_score desc)
        pairs = list(freq.items())  # [(pair, count), ...]
        pairs.sort(
            key=lambda kv: (kv[1], statistics.mean(scores_for_pair[kv[0]])),
            reverse=True,
        )
        best_pair = pairs[0][0]  # (pid_offset, pskip)
        mapping[fname] = best_pair

    return mapping


# ======================================================================
# Main ==========================================================================
# ======================================================================

mapping = build_offset_map(data_dir="data_raw", n_packets_per_file=10)
print("Per-file mapping:")
for fname, (pid, pskip) in mapping.items():
    print(f"  {fname}: pid_offset={pid}, post_pid_skip={pskip}")
