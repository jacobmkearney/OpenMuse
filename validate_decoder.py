import os
from typing import Optional, Dict, List, Any, Tuple
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as _dt
import struct

# STARTING INFO ====================================================================


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

# Different presets may enable/disable some channels.

# Note: 1034 is the only preset for which a red LED brightly turned on during recording, suggesting the activation of OPTICS or PPG channels.

# Based on these specs, we can derive the following plausible expectations regarding each channel type:


EXPECTED_RATES = {
    "EEG": 256.0,
    "AUX": 256.0,
    "ACC": 52.0,
    "GYRO": 52.0,
    "PPG": 64.0,
    "OPTICS": 64.0,
}

EXPECTED_CHANNELS = {
    "EEG": [0, 4],  #  256 Hz, 14 bits
    "AUX": [0, 1, 4],  # 256 Hz, 14 bits
    "ACC": [0, 3],  #  52 Hz, 16 bits
    "GYRO": [0, 3],  #  52 Hz, 16 bits
    "PPG": [0, 3],  # 64 Hz, 20 bits
    "OPTICS": [0, 1, 4, 5, 8, 16],  # 64 Hz, 20 bits
}

# Each data file (./data_raw/) should contain some combination of these channels.
# The exact combination depends on the preset used during recording.
# Importantly, these channels types are likely indistinguishable from the data alone, so it is best to group them according to their data characteristics.

EXPECTED_GROUPS = {
    "CH256": set(
        EXPECTED_CHANNELS["EEG"]
        + EXPECTED_CHANNELS["AUX"]
        + [
            i + j
            for i in EXPECTED_CHANNELS["EEG"]
            for j in EXPECTED_CHANNELS["AUX"]
            if j > 0
        ]
    ),
    "CH52": set(
        EXPECTED_CHANNELS["ACC"]
        + EXPECTED_CHANNELS["GYRO"]
        + [
            i + j
            for i in EXPECTED_CHANNELS["ACC"]
            for j in EXPECTED_CHANNELS["GYRO"]
            if j > 0
        ]
    ),
    "CH64": set(
        EXPECTED_CHANNELS["PPG"]
        + EXPECTED_CHANNELS["OPTICS"]
        + [
            i + j
            for i in EXPECTED_CHANNELS["PPG"]
            for j in EXPECTED_CHANNELS["OPTICS"]
            if j > 0
        ]
    ),
}

EXPECTED_BITS = {
    "CH256": 14,
    "CH52": 16,
    "CH64": 20,
}

# ------------------------------------------------------------------------------

# Goals:
# - 1) Decode raw packets into structured data with channels labeled according to their type (group) and index.
# - 2) For each preset (i.e., each file), infer the most likely configuration of channels and groups based on the data.
# - 3) Make the decoding logic data-driven, naturally flowing from the prior knowledge about the channel counts and rates and little else.

# Logic: Iterate through each file (i.e., each preset which might contain a different combination of active channels), and through each possible combination of channels based on EXPECTED_GROUPS (e.g., 4 CH256 + 0 CH52 + 3 CH64, etc.). For each instance, try various decoding strategies. Collect the result, for each file, each combination, and each decoding strategies in a pandas DataFrame. Based on the results, try to infer what is the best decoding strategy and channel combination for each file.


# DECODING ====================================================================


def decode_muse_method1(
    line: bytes | str, timestamp: Optional[_dt.datetime] = None
) -> Dict[str, Any]:
    """Decoding method based on Amused-py."""

    # --- Constants ---
    CH256_SEG_BYTES = 18
    CH256_PAIR_BYTES = 3
    CH256_CENTER = 2048
    CH256_SAMPLE_MAX = 4095

    CH64_20BIT_PAIR_BYTES = 5
    CH64_PLAUSIBLE_MIN = 10000

    CH52_TUPLE_BYTES = 12

    # --- Minimal helpers ---
    def _unpack_ch256_18b(seg: bytes):
        out = []
        for i in range(CH256_SEG_BYTES // CH256_PAIR_BYTES):
            b0, b1, b2 = seg[i * 3 : i * 3 + 3]
            s1 = (b0 << 4) | (b1 >> 4)
            s2 = ((b1 & 0x0F) << 8) | b2
            out.extend([s1 - CH256_CENTER, s2 - CH256_CENTER])
        return out

    def _looks_like_ch256(seg: bytes) -> bool:
        if len(seg) != CH256_SEG_BYTES:
            return False
        sample = (seg[0] << 4) | (seg[1] >> 4)
        return 0 <= sample <= CH256_SAMPLE_MAX

    def _unpack_ch64_pair(b: bytes, off: int):
        if off + CH64_20BIT_PAIR_BYTES > len(b):
            return None
        b0, b1, b2, b3, b4 = b[off : off + 5]
        v1 = (b0 << 12) | (b1 << 4) | (b2 >> 4)
        v2 = ((b2 & 0x0F) << 16) | (b3 << 8) | b4
        return v1, v2

    def _extract_ch52_bulk(data: bytes, start_min=4, min_samples=2):
        n = len(data)
        best = []
        for base in range(start_min, n - CH52_TUPLE_BYTES):
            off = base
            items = []
            while off + CH52_TUPLE_BYTES <= n:
                try:
                    ax, ay, az, gx, gy, gz = struct.unpack_from(">hhhhhh", data, off)
                except Exception:
                    break
                items.append({"accel": [ax, ay, az], "gyro": [gx, gy, gz]})
                off += CH52_TUPLE_BYTES
            if len(items) > len(best):
                best = items
        return best if len(best) >= min_samples else []

    # --- Input handling ---
    if isinstance(line, str):
        data = bytes.fromhex(line)
    else:
        data = line
    if not data:
        return {"timestamp": timestamp or _dt.datetime.now(), "packet_type": "EMPTY"}

    b0 = data[0]
    result: Dict[str, Any] = {
        "timestamp": timestamp or _dt.datetime.now(),
        "packet_type": f"0x{b0:02X}",
    }

    # CH256 bulk frames
    if b0 in (0xDF, 0xE2, 0xE5, 0xEE, 0xEF, 0xF2, 0xD9, 0xDB, 0xCF, 0xCA, 0xCB, 0xCE):
        res = {"ch256": {}, "ch64": {}}
        offset = 4
        ch_idx = 0
        while offset + CH256_SEG_BYTES <= len(data):
            seg = data[offset : offset + 18]
            if _looks_like_ch256(seg):
                samples = _unpack_ch256_18b(seg)
                key = f"CH256_{ch_idx+1}"
                res["ch256"].setdefault(key, []).extend(samples)
                ch_idx += 1
                offset += 18
            else:
                offset += 1
        result.update(res)

    # CH64 packets
    elif b0 in (0xE3, 0xEC, 0xF0):
        res = {"ch64": {"CH64_1": [], "CH64_2": [], "CH64_3": []}}
        offset, n = 4, len(data)
        idx = 0
        while offset + 5 <= n:
            v = _unpack_ch64_pair(data, offset)
            if not v:
                break
            v1, v2 = v
            if all(CH64_PLAUSIBLE_MIN <= x < (1 << 20) for x in (v1, v2)):
                ch1, ch2 = idx % 3, (idx + 1) % 3
                res["ch64"][f"CH64_{ch1+1}"].append(v1)
                res["ch64"][f"CH64_{ch2+1}"].append(v2)
                idx += 2
                offset += 5
            else:
                break
        if idx >= 6:
            result.update(res)

    # CH52 packets
    elif b0 in (0xF4, 0xDA):
        try:
            ax, ay, az, gx, gy, gz = struct.unpack_from(">hhhhhh", data, 4)
            result["ch52"] = {"ACC": [ax, ay, az], "GYRO": [gx, gy, gz]}
        except Exception:
            pass
    elif b0 in (0xD7, 0xD1, 0xD5, 0xDD):
        batch = _extract_ch52_bulk(data, start_min=4)
        if batch:
            result["ch52_batch"] = batch

    return result


def _score_rates(inferred: Dict[str, float], expected: Dict[str, float]) -> float:
    """Score inferred rates against expected rates.

    Missing rates are treated as a finite penalty (100% relative error),
    so this function never returns inf.
    """
    s = 0.0
    count = 0
    for k, exp in expected.items():
        inf = inferred.get(k)
        if inf is None:
            # treat missing rate as a full relative error (1.0)
            rel = 1.0
        else:
            rel = (inf - exp) / exp
        s += rel * rel
        count += 1
    return s / max(1, count)


def _generate_config_candidates() -> List[Dict[str, int]]:
    """Generate candidate configurations at the GROUP level only.

    Returns a list of dicts with keys: 'CH256','CH52','CH64'.
    """
    ch256_choices = sorted(EXPECTED_GROUPS.get("CH256", []))
    ch52_choices = sorted(EXPECTED_GROUPS.get("CH52", []))
    ch64_choices = sorted(EXPECTED_GROUPS.get("CH64", []))

    candidates: List[Dict[str, int]] = []
    for g256, g52, g64 in itertools.product(ch256_choices, ch52_choices, ch64_choices):
        candidates.append({"CH256": g256, "CH52": g52, "CH64": g64})
    return candidates


def analyze_file_with_config(
    lines: List[str], config: Optional[Dict[str, int]] = None
) -> Dict[str, Any]:
    stats = {
        "lines": 0,
        "decoded_lines": 0,
        "errors": 0,
        "duration_s": 0.0,
        "rates": {},
        "ch256_samples": 0,
        "ch52_samples": 0,
        "ch64_samples": 0,
    }
    times: List[float] = []
    config = config or {"CH256": 0, "CH52": 0, "CH64": 0}
    expected_tokens = (
        config.get("CH256", 0) + config.get("CH52", 0) + config.get("CH64", 0)
    )

    for line in lines:
        stats["lines"] += 1
        s = line.strip()
        if not s:
            continue
        parts = [p for p in (s.split(",") if "," in s else s.split()) if p]
        t = None
        try:
            if parts and ("." in parts[0] or parts[0].isdigit()):
                t = float(parts[0])
                parts = parts[1:]  # Remove timestamp
        except Exception:
            stats["errors"] += 1
            continue
        if t is not None:
            times.append(t)

        # Validate token count against config
        num_tokens = len(parts)
        if num_tokens == expected_tokens:
            stats["decoded_lines"] += 1
            # Assign tokens to groups based on config and bit resolution
            token_idx = 0
            for group, count in config.items():
                if count == 0:
                    continue
                expected_bits = EXPECTED_BITS.get(group, 14)
                for _ in range(count):
                    if token_idx >= num_tokens:
                        stats["errors"] += 1
                        break
                    try:
                        val = float(parts[token_idx])
                        # Check if value is within expected bit resolution range
                        max_val = 2**expected_bits - 1
                        if abs(val) > max_val:
                            stats["errors"] += 1
                        else:
                            stats[f"{group.lower()}_samples"] += 1
                    except ValueError:
                        stats["errors"] += 1
                    token_idx += 1
        else:
            stats["errors"] += 1

    # Rate estimation
    if len(times) >= 2:
        try:
            duration = max(times) - min(times)
            stats["duration_s"] = float(duration) if duration > 0 else 0.0
            for group in ["CH256", "CH52", "CH64"]:
                samples = stats.get(f"{group.lower()}_samples", 0)
                if stats["duration_s"] > 0 and samples > 0:
                    stats["rates"][group] = samples / stats["duration_s"]
        except Exception:
            stats["errors"] += 1

    return stats


def run_search_on_file(lines: List[str], file_label: str):
    """Try candidate channel configs and decoding strategies for one file.

    Returns a pandas.DataFrame-like list of records with scores and inferred rates.
    """
    candidates = _generate_config_candidates()
    records: List[Dict[str, Any]] = []

    for grp_cfg in candidates:
        stats = analyze_file_with_config(lines, config=grp_cfg)
        rates = stats.get("rates", {})
        rate_score = _score_rates(
            rates,
            {
                "CH256": EXPECTED_RATES.get("EEG", 256.0),
                "CH64": EXPECTED_RATES.get("PPG", 64.0),
                "CH52": EXPECTED_RATES.get("ACC", 52.0),
            },
        )
        # Mismatch based on sample counts
        duration = stats.get("duration_s", 0.0)
        mismatch256 = 0.0
        if duration > 0:
            expected_samples_256 = (
                grp_cfg.get("CH256", 0) * EXPECTED_RATES["EEG"] * duration
            )
            detected_samples_256 = stats.get("ch256_samples", 0)
            mismatch256 = (detected_samples_256 - expected_samples_256) / max(
                1, expected_samples_256
            )
        mismatch_sq256 = mismatch256 * mismatch256

        mismatch52 = 0.0
        if duration > 0:
            expected_samples_52 = (
                grp_cfg.get("CH52", 0) * EXPECTED_RATES["ACC"] * duration
            )
            detected_samples_52 = stats.get("ch52_samples", 0)
            mismatch52 = (detected_samples_52 - expected_samples_52) / max(
                1, expected_samples_52
            )
        mismatch_sq52 = mismatch52 * mismatch52

        mismatch64 = 0.0
        if duration > 0:
            expected_samples_64 = (
                grp_cfg.get("CH64", 0) * EXPECTED_RATES["PPG"] * duration
            )
            detected_samples_64 = stats.get("ch64_samples", 0)
            mismatch64 = (detected_samples_64 - expected_samples_64) / max(
                1, expected_samples_64
            )
        mismatch_sq64 = mismatch64 * mismatch64

        err_pen = stats.get("errors", 0) / max(1, stats.get("lines", 1))
        score = (
            rate_score
            + 1.0 * mismatch_sq256
            + 1.0 * mismatch_sq52
            + 1.0 * mismatch_sq64
            + 5.0 * err_pen
        )

        rec = {
            "file": file_label,
            "group_config": grp_cfg,
            "score": score,
            "decoded_lines": stats.get("decoded_lines", 0),
            "errors": stats.get("errors", 0),
            "duration_s": stats.get("duration_s", 0.0),
            "rates": rates,
        }

        records.append(rec)

    # Expand rates into columns for sorting
    rows = []
    for r in records:
        row = {
            "file": r["file"],
            "score": r["score"],
            "decoded_lines": r["decoded_lines"],
            "errors": r["errors"],
            "duration_s": r["duration_s"],
        }
        # Flatten group config
        grp = r.get("group_config") or {}
        row["grp_CH256"] = grp.get("CH256")
        row["grp_CH52"] = grp.get("CH52")
        row["grp_CH64"] = grp.get("CH64")
        # Flatten only District's rates (CH256, CH52, CH64)
        for k in ("CH256", "CH52", "CH64"):
            v = r.get("rates", {}).get(k)
            if v is not None:
                row[f"rate_{k}"] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.sort_values(["score"]) if not df.empty else df


def perfectness_score(df: pd.DataFrame) -> Dict[str, float]:
    """Compute a small summary that indicates how 'peaked' the scoring is.

    Metrics returned:
    - best_score: the lowest (best) score
    - gap_ratio: (second_best - best) / max(1e-12, abs(best)) — larger is better
    - top_fraction: fraction of total inverse-score mass concentrated in the top candidate
    - n_candidates: number of candidates considered
    """
    if df is None or df.empty:
        return {
            "best_score": float("inf"),
            "gap_ratio": 0.0,
            "top_fraction": 0.0,
            "n_candidates": 0,
        }
    scores = df["score"].to_numpy()
    # ignore non-finite scores
    import numpy as _np

    finite_mask = _np.isfinite(scores)
    if not finite_mask.any():
        return {
            "best_score": float("inf"),
            "gap_ratio": 0.0,
            "top_fraction": 0.0,
            "n_candidates": int(len(scores)),
        }
    scores = scores[finite_mask]
    # lower is better -> convert to positive affinities
    # handle non-positive or zero by adding offset
    min_score = float(scores.min())
    sorted_idx = scores.argsort()
    best = float(scores[sorted_idx[0]])
    n = len(scores)
    second = float(scores[sorted_idx[1]]) if n > 1 else best
    gap_ratio = (second - best) / (abs(best) + 1e-12)

    # convert to inverse scores for soft-weights (higher is better)
    inv = 1.0 / (scores - best + 1e-6)
    total = float(inv.sum())
    # sorted_idx refers to indices into the original scores array; map to finite-only
    # recompute sorted order for the finite scores
    f_sorted_idx = scores.argsort()
    top_fraction = float(inv[f_sorted_idx[0]] / total) if total > 0 else 0.0
    return {
        "best_score": best,
        "gap_ratio": gap_ratio,
        "top_fraction": top_fraction,
        "n_candidates": n,
    }


# RUN ===============================================================================

if __name__ == "__main__":
    files = [f for f in os.listdir("data_raw") if f.endswith(".txt")]

    results = []
    for path in files:
        full_path = os.path.join("data_raw", path)
        with open(full_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        print(f"Processing {path}... (this may take a moment)")
        try:
            df: pd.DataFrame = run_search_on_file(lines, file_label=path)
        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue
        if df is not None and not df.empty:
            # compute perfectness summary for this file and annotate each row
            summary = perfectness_score(df)
            for k, v in summary.items():
                df[f"summary_{k}"] = v
            results.append(df)
    if not results:
        print("No results to plot.")
    else:
        all_df = pd.concat(results, ignore_index=True)
        all_df = all_df.reset_index(drop=True)
        all_df["config"] = (
            all_df["grp_CH256"].astype(str)
            + "-"
            + all_df["grp_CH52"].astype(str)
            + "-"
            + all_df["grp_CH64"].astype(str)
        )

        # Visualize results for score
        import seaborn as sns

        # Normalize scores within each file
        all_df["normalized_score"] = all_df.groupby("file")["score"].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-12)
        )
        pivot = all_df.pivot_table(
            index="file", columns="config", values="normalized_score", aggfunc="min"
        )
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot)
        plt.title("Normalized Score Heatmap")
        plt.show()
