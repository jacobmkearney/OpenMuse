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
    "CH256": list(
        set(
            EXPECTED_CHANNELS["EEG"]
            + EXPECTED_CHANNELS["AUX"]
            + [
                i + j
                for i in EXPECTED_CHANNELS["EEG"]
                for j in EXPECTED_CHANNELS["AUX"]
                if j > 0
            ]
        )
    ),
    "CH52": list(
        set(
            EXPECTED_CHANNELS["ACC"]
            + EXPECTED_CHANNELS["GYRO"]
            + [
                i + j
                for i in EXPECTED_CHANNELS["ACC"]
                for j in EXPECTED_CHANNELS["GYRO"]
                if j > 0
            ]
        )
    ),
    "CH64": list(
        set(
            EXPECTED_CHANNELS["PPG"]
            + EXPECTED_CHANNELS["OPTICS"]
            + [
                i + j
                for i in EXPECTED_CHANNELS["PPG"]
                for j in EXPECTED_CHANNELS["OPTICS"]
                if j > 0
            ]
        )
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
    data: bytes | str,
    timestamp: Optional[_dt.datetime] = None,
    config: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """
    This version integrates:
    1.  Smart alignment search for CH256 (EEG) data, handling interleaved packets.
    2.  Demultiplexing of CH64 (PPG) data into three distinct channels.
    3.  Use of the 'config' parameter to determine CH256 channel grouping.

    """
    res = {
        "timestamp": timestamp or _dt.datetime.now(),
        "ok": False,
        "errors": 0,
        "ch256_samples": 0,
        "ch52_samples": 0,
        "ch64_samples": 0,
    }

    # --- constants ---
    CH256_SEG_BYTES = 18
    CH256_PAIR_BYTES = 3
    CH256_SAMPLE_MAX = 4095
    CH256_CENTER = 2048
    MIN_ALIGNED_RUN = 2

    CH64_20BIT_PAIR_BYTES = 5
    CH64_PLAUSIBLE_MIN = 10000

    CH52_TUPLE_BYTES = 12
    CH52_DEFAULT_START_MIN = 4
    CH52_MIN_SAMPLES = 2

    # --- helpers ---
    def _looks_like_ch256(seg: bytes) -> bool:
        if len(seg) != CH256_SEG_BYTES:
            return False
        sample = (seg[0] << 4) | (seg[1] >> 4)
        return 0 <= sample <= CH256_SAMPLE_MAX

    def _unpack_ch256_18b(seg: bytes) -> List[int]:
        out = []
        for i in range(CH256_SEG_BYTES // CH256_PAIR_BYTES):
            b0, b1, b2 = seg[i * 3 : i * 3 + 3]
            s1 = (b0 << 4) | (b1 >> 4)
            s2 = ((b1 & 0x0F) << 8) | b2
            out.extend([s1 - CH256_CENTER, s2 - CH256_CENTER])
        return out

    def _unpack_ch64_pair(b: bytes, off: int) -> Optional[Tuple[int, int]]:
        if off + CH64_20BIT_PAIR_BYTES > len(b):
            return None
        b0, b1, b2, b3, b4 = b[off : off + 5]
        v1 = (b0 << 12) | (b1 << 4) | (b2 >> 4)
        v2 = ((b2 & 0x0F) << 16) | (b3 << 8) | b4
        return v1, v2

    def _best_ch256_alignment(data: bytes, start_min: int) -> Tuple[int, int]:
        n = len(data)
        best_off, best_run = -1, 0
        for base in range(start_min, n - CH256_SEG_BYTES + 1):
            off, run = base, 0
            while off + CH256_SEG_BYTES <= n and _looks_like_ch256(
                data[off : off + CH256_SEG_BYTES]
            ):
                run += 1
                off += CH256_SEG_BYTES
            if run > best_run:
                best_run, best_off = run, base
        return best_off, best_run

    def _extract_ch52_bulk(data: bytes) -> int:
        n = len(data)
        best_run = 0
        for base in range(CH52_DEFAULT_START_MIN, n - CH52_TUPLE_BYTES + 1):
            count = 0
            off = base
            while off + CH52_TUPLE_BYTES <= n:
                try:
                    struct.unpack_from(">hhhhhh", data, off)
                    count += 1
                    off += CH52_TUPLE_BYTES
                except struct.error:
                    break
            if count > best_run:
                best_run = count
        return best_run if best_run >= CH52_MIN_SAMPLES else 0

    # --- main decode ---
    try:
        data = bytes.fromhex(data) if isinstance(data, str) else data
        if not data:
            res["errors"] += 1
            return res

        b0 = data[0]

        # CH256
        if b0 in {
            0xDF,
            0xE2,
            0xE5,
            0xEE,
            0xEF,
            0xF2,
            0xD9,
            0xDB,
            0xCF,
            0xCA,
            0xCB,
            0xCE,
        }:
            offset = 4
            while offset < len(data):
                base, run = _best_ch256_alignment(data, start_min=offset)
                if base == -1 or run < MIN_ALIGNED_RUN:
                    offset += 1
                    continue
                current_pos = base
                while current_pos + CH256_SEG_BYTES <= len(data):
                    seg = data[current_pos : current_pos + CH256_SEG_BYTES]
                    if not _looks_like_ch256(seg):
                        break
                    samples = _unpack_ch256_18b(seg)
                    res["ch256_samples"] += len(samples)
                    current_pos += CH256_SEG_BYTES
                offset = current_pos + CH256_SEG_BYTES

        # CH64
        elif b0 in {0xE3, 0xEC, 0xF0}:
            offset = 4
            while offset + CH64_20BIT_PAIR_BYTES <= len(data):
                pair = _unpack_ch64_pair(data, offset)
                if not pair:
                    break
                v1, v2 = pair
                if all(CH64_PLAUSIBLE_MIN <= v < (1 << 20) for v in (v1, v2)):
                    res["ch64_samples"] += 2
                    offset += CH64_20BIT_PAIR_BYTES
                else:
                    break

        # CH52 single
        elif b0 in {0xF4, 0xDA}:
            try:
                struct.unpack_from(">hhhhhh", data, 4)
                res["ch52_samples"] += 1
            except struct.error:
                res["errors"] += 1

        # CH52 bulk
        elif b0 in {0xD7, 0xD1, 0xD5, 0xDD}:
            res["ch52_samples"] += _extract_ch52_bulk(data)

    except Exception:
        res["errors"] += 1
        return res

    if any((res["ch256_samples"], res["ch52_samples"], res["ch64_samples"])):
        res["ok"] = True

    return res


def decode_muse_method2(
    data: bytes | str,
    timestamp: Optional[_dt.datetime] = None,
    config: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Token-based decoding extracted from `analyze_file_with_config`.

    Returns a standardized dict with keys: 'timestamp','ok','errors',
    'ch256_samples','ch52_samples','ch64_samples'.
    """
    res = {
        "timestamp": timestamp or _dt.datetime.now(),
        "ok": False,
        "errors": 0,
        "ch256_samples": 0,
        "ch52_samples": 0,
        "ch64_samples": 0,
    }
    # normalize to str
    if isinstance(data, (bytes, bytearray)):
        try:
            s = data.decode("utf-8", errors="ignore").strip()
        except Exception:
            s = str(data).strip()
    else:
        s = str(data).strip()
    if not s:
        res["errors"] += 1
        return res

    parts = [p for p in (s.split(",") if "," in s else s.split()) if p]
    t = None
    try:
        if parts and ("." in parts[0] or parts[0].isdigit()):
            t = float(parts[0])
            parts = parts[1:]
    except Exception:
        res["errors"] += 1
        return res
    if t is not None:
        res["timestamp"] = t

    config = config or {"CH256": 0, "CH52": 0, "CH64": 0}
    expected_tokens = (
        config.get("CH256", 0) + config.get("CH52", 0) + config.get("CH64", 0)
    )
    num_tokens = len(parts)
    if num_tokens != expected_tokens:
        res["errors"] += 1
        return res

    # assign tokens
    token_idx = 0
    for group, count in config.items():
        if count == 0:
            continue
        expected_bits = EXPECTED_BITS.get(group, 14)
        for _ in range(count):
            if token_idx >= num_tokens:
                res["errors"] += 1
                break
            try:
                val = float(parts[token_idx])
                max_val = 2**expected_bits - 1
                if abs(val) > max_val:
                    res["errors"] += 1
                else:
                    if group == "CH256":
                        res["ch256_samples"] += 1
                    elif group == "CH52":
                        res["ch52_samples"] += 1
                    elif group == "CH64":
                        res["ch64_samples"] += 1
            except ValueError:
                res["errors"] += 1
            token_idx += 1

    if res["errors"] == 0:
        res["ok"] = True
    return res


METHODS = {"method1": decode_muse_method1, "method2": decode_muse_method2}

# VALIDATION ====================================================================


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
    candidates: List[Dict[str, int]] = []
    for g256, g52, g64 in itertools.product(
        EXPECTED_GROUPS["CH256"], EXPECTED_GROUPS["CH52"], EXPECTED_GROUPS["CH64"]
    ):
        candidates.append({"CH256": g256, "CH52": g52, "CH64": g64})
    return candidates


def analyze_file_with_config(
    lines: List[str], config: Optional[Dict[str, int]] = None, decode_fn=None
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

    decode_fn = decode_fn or decode_muse_method1
    for line in lines:
        stats["lines"] += 1

        # Unpack line: each line has three tab‑separated fields:
        # - Timestamp (ISO 8601 with microseconds and timezone)
        # - UUID / session ID
        # - Hex payload (the actual Muse packet)
        parts = line.strip().split("\t")
        if len(parts) != 3:
            print(f"Malformed line: {line}")
            continue
        ts_str, uuid_str, hex_payload = parts[0], parts[1], parts[2]

        # Parse
        ts = _dt.datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        try:
            dec = decode_fn(hex_payload, timestamp=ts, config=config)
        except Exception:
            stats["errors"] += 1
            continue

        # accumulate
        times.append(ts.timestamp())

        stats["errors"] += dec.get("errors", 0)
        if dec.get("ok"):
            stats["decoded_lines"] += 1
        stats["ch256_samples"] += dec.get("ch256_samples", 0)
        stats["ch52_samples"] += dec.get("ch52_samples", 0)
        stats["ch64_samples"] += dec.get("ch64_samples", 0)

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


def run_search_on_file(
    lines: List[str], file_label: str, method="method1"
) -> pd.DataFrame:
    """Try candidate channel configs and decoding strategies for one file.

    Returns a pandas.DataFrame-like list of records with scores and inferred rates.
    """
    candidates = _generate_config_candidates()
    records: List[Dict[str, Any]] = []

    for grp_cfg in candidates:
        stats = analyze_file_with_config(
            lines, config=grp_cfg, decode_fn=METHODS[method]
        )
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
            "method": method,
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
            "method": r["method"],
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

        for method in METHODS.keys():
            print(f"  Using decoding method: {method}")
            try:
                df: pd.DataFrame = run_search_on_file(
                    lines, file_label=path, method=method
                )
            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue
            if df is not None and not df.empty:
                # compute perfectness summary for this file and annotate each row
                summary = perfectness_score(df)
                for k, v in summary.items():
                    df[f"summary_{k}"] = v
                results.append(df)

    # Evaluate
    if not results:
        print("No results to plot.")
    else:
        rez = pd.concat(results, ignore_index=True)
        rez = rez.reset_index(drop=True)
        rez["config"] = (
            rez["grp_CH256"].astype(str)
            + "-"
            + rez["grp_CH52"].astype(str)
            + "-"
            + rez["grp_CH64"].astype(str)
        )

        # Visualize results for score
        import seaborn as sns
        import matplotlib.pyplot as plt

        rez["score_log"] = np.log1p(rez["score"])

        for method, subdf in rez.groupby("method"):
            pivot = subdf.pivot_table(
                index="file", columns="config", values="score_log", aggfunc="min"
            )
            plt.figure(figsize=(12, 8))
            sns.heatmap(pivot)
            plt.title(f"Normalized Score Heatmap – {method}")
            plt.gca().collections[0].set_clim(rez["score_log"].min(), rez["score_log"].max())
            plt.show()
