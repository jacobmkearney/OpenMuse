"""
Channel Header Analysis Script
------------------------------

Purpose:
- Explore raw Muse data files (.txt) to infer which packet headers correspond
  to which channel groups (CH256, CH52, CH64).
- Use timestamps to compute packet rates, then infer samples per packet (spp).
- Compare inferred spp to expected canonical values (≈12, 6, 2).
- Classify headers accordingly and produce a summary table per file.

Key Ideas:
- Each header byte likely corresponds to a packet type.
- If a header appears N times per second, then to deliver a known sampling rate R,
  each packet must carry roughly R/N samples.
- For CH256, CH52, CH64, this tends to yield integers near 12, 6, 2 respectively.

Note:
- The packets are not fixed in any way, they contain a random assortment of subpackets, so it might be difficult to further figure out what they contain just with raw pattern analysis.
"""

import os
import datetime as dt
from collections import defaultdict
import numpy as np
import pandas as pd
from functools import reduce
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from pathlib import Path
from typing import Sequence

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


# --- Allowed multiples to check ---
MULTIPLES = [1, 2]


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


def packet_frequency(times):
    """Packets per second from timestamps; robust via median inter-arrival."""
    if len(times) < 2:
        return None
    ts = np.sort(times)
    diffs = np.diff(ts)
    med = np.median(diffs)
    return 1.0 / med if med > 0 else None


# ADVANCED DECODER FUNCTIONS ============================================================
def estimate_overhead_bytes(payloads, max_overhead=8):
    """
    Estimate overhead bytes for a header by maximizing divisibility and byte-alignment
    over the observed payload lengths.

    Returns
    -------
    (best_overhead, score, diagnostics)
    """
    lens = [len(p) for p in payloads if len(p) > 0]
    if not lens:
        return None, 0.0, {"lens": []}

    lens = np.array(lens, dtype=int)
    med = int(np.median(lens))
    mode_len = int(pd.Series(lens).mode().iloc[0]) if len(lens) else med

    plausible_bps = [14, 16, 20]
    plausible_ch = list(range(1, 17))

    def score_overhead(ob):
        data_bytes = lens - ob
        if np.any(data_bytes <= 0):
            return -np.inf, {}
        data_bits = data_bytes * 8

        align_hits = 0
        remainder_score = 0.0
        for bps in plausible_bps:
            for ch in plausible_ch:
                denom = bps * ch
                remainders = data_bits % denom
                align_hits += np.sum(remainders == 0)
                remainder_score += np.sum(1.0 / (1.0 + remainders))

        unique_data = np.unique(data_bytes)
        stability = 1.0 / len(unique_data)

        prior = 1.0 - min(1.0, abs(ob - 3) / 5.0)

        total_cases = len(lens) * len(plausible_bps) * len(plausible_ch)
        score = (
            0.50 * (align_hits / total_cases)
            + 0.30 * (remainder_score / total_cases)
            + 0.15 * stability
            + 0.05 * prior
        )
        diag = {
            "unique_lengths": unique_data.tolist(),
            "mode_len": mode_len,
            "median_len": med,
        }
        return score, diag

    candidates = []
    for ob in range(0, max_overhead + 1):
        s, d = score_overhead(ob)
        candidates.append((s, ob, d))

    candidates.sort(key=lambda x: (-x[0], x[1]))  # best score, then smaller ob
    best_score, best_ob, diagnostics = candidates[0]
    return best_ob, best_score, diagnostics


def detect_header_cycle(headers, max_period=16):
    """
    Detect a repeating cycle by position-wise purity over candidate periods.
    headers: list of header bytes in arrival order.
    Returns (best_period, purity, position_profiles)
    """
    import collections

    seq = [h for h in headers if h is not None]
    if len(seq) < 50:
        return None, 0.0, []

    best = (None, 0.0, [])
    for p in range(2, max_period + 1):
        buckets = [collections.Counter() for _ in range(p)]
        for i, h in enumerate(seq):
            buckets[i % p][h] += 1
        # Purity: how dominant is the top header in each bucket
        purities = []
        profiles = []
        for b in buckets:
            total = sum(b.values())
            if total == 0:
                purities.append(0.0)
                profiles.append({})
            else:
                top_hdr, top_cnt = max(b.items(), key=lambda kv: kv[1])
                purities.append(top_cnt / total)
                profiles.append(dict(b))
        mean_purity = float(np.mean(purities))
        if mean_purity > best[1]:
            best = (p, mean_purity, profiles)
    return best


def sequence_groups(headers, period, profiles, min_share=0.5):
    """
    Map headers to dominant cycle positions. Returns dict:
    Header -> {"positions": [k...], "support": share}
    """
    pos_map = {}
    for k, prof in enumerate(profiles):
        total = sum(prof.values())
        if total == 0:
            continue
        # headers with decent share at this position
        for hdr, cnt in prof.items():
            share = cnt / total
            if share >= min_share:
                pos_map.setdefault(hdr, {"positions": set(), "support": 0.0})
                pos_map[hdr]["positions"].add(k)
                pos_map[hdr]["support"] = max(pos_map[hdr]["support"], share)
    # finalize sets to lists
    for hdr in pos_map:
        pos_map[hdr]["positions"] = sorted(list(pos_map[hdr]["positions"]))
    return pos_map


def sequence_rate_hint(times, period, headers, target_hdr):
    """
    Estimate per-header packet rate from cycle period and arrivals.
    """
    base_hz = packet_frequency(times) or 0.0
    # Count occurrences of target_hdr per cycle mod period
    positions = []
    for i, h in enumerate(headers):
        if h == target_hdr:
            positions.append(i % period)
    occ_per_cycle = len(set(positions)) if positions else 0
    if period and occ_per_cycle:
        return base_hz * (occ_per_cycle / period)
    return None


def classify_header(
    tlist,
    payloads,
    tol_blocks=0.15,
    tol_rate=0.20,
    overhead_fn=None,
    note_tags=None,
    pkt_hz_hint=None,
):
    """
    Classify a packet header into CH256/CH52/CH64.

    Parameters
    ----------
    tlist : list of float
        Packet timestamps (seconds).
    payloads : list of bytes
        Raw payloads for a given header.
    tol_blocks : float
        Tolerance for integer blocks match.
    tol_rate : float
        Tolerance for observed sampling rate match.
    overhead_fn : callable or None
        Function(payloads) -> (overhead_bytes, score, diagnostics).
        If provided and successful, use that overhead; else fall back to [2,3,4].
    note_tags : list of str or None
        Extra annotations to include in the returned 'note' field.
    pkt_hz_hint : float or None
        If provided, use this packet frequency (Hz) instead of computing from tlist.
        Intended for blending sequence-level hints with raw timing.

    Returns
    -------
    dict or None
        Classification result with keys: ch, num_ch, blocks, obs_hz, err, note.
    """
    BASE_SPP = {"CH256": 12, "CH52": 6, "CH64": 2}

    # Use hint if available; otherwise compute
    pkt_hz = (
        pkt_hz_hint
        if (pkt_hz_hint is not None and pkt_hz_hint > 0)
        else packet_frequency(tlist)
    )
    if not pkt_hz:
        return None

    lengths = [len(p) for p in payloads if len(p) > 0]
    if not lengths:
        return None
    median_len = int(np.median(lengths))

    # Determine overhead candidates
    extra_note = []
    overhead_candidates = [2, 3, 4]  # default fallback
    if callable(overhead_fn):
        ob, score, diag = overhead_fn(payloads)
        if ob is not None:
            overhead_candidates = [int(ob)]
            extra_note.append(f"overhead_fn={int(ob)},score={score:.3f}")
        else:
            extra_note.append("overhead_fn_failed")

    best = {
        "ch": "uncertain",
        "num_ch": None,
        "blocks": None,
        "obs_hz": None,
        "err": np.inf,
        "note": None,
    }

    # --- 1. Single-group classification using payload math ---
    for overhead_bytes in overhead_candidates:
        data_bytes = max(median_len - overhead_bytes, 0)
        median_bits = data_bytes * 8
        if median_bits <= 0:
            continue

        for ch in EXPECTED_GROUPS:
            rate = EXPECTED_RATES[ch]
            base = BASE_SPP[ch]
            for num_ch in EXPECTED_GROUPS[ch]:
                if num_ch <= 0:
                    continue
                for bps in [BITS_PER_SAMPLE[ch], 16, 24]:  # canonical + alternates
                    denom = num_ch * bps
                    if denom <= 0:
                        continue

                    spp_size = median_bits / denom
                    if spp_size <= 0:
                        continue

                    blocks = spp_size / base
                    blocks_int = int(round(blocks))
                    if abs(blocks - blocks_int) > tol_blocks:
                        continue

                    obs_hz = pkt_hz * spp_size
                    rel_err = abs(obs_hz - rate) / rate if rate > 0 else np.inf
                    if rel_err < tol_rate and rel_err < best["err"]:
                        best = {
                            "ch": ch,
                            "num_ch": num_ch,
                            "blocks": blocks_int,
                            "obs_hz": obs_hz,
                            "err": rel_err,
                            "note": f"bps={bps},overhead={overhead_bytes}",
                        }

        if best["ch"] != "uncertain":
            break

    # --- 2. Timing-only fallback ---
    if best["ch"] == "uncertain":
        for ch in EXPECTED_GROUPS:
            rate = EXPECTED_RATES[ch]
            spp_est = rate / pkt_hz if pkt_hz > 0 else 0
            for mult in MULTIPLES:
                base_mult = BASE_SPP[ch] * mult
                blocks = spp_est / base_mult
                blocks_int = int(round(blocks))
                if blocks_int > 0 and abs(blocks - blocks_int) < tol_blocks:
                    obs_hz = pkt_hz * (blocks_int * base_mult)
                    rel_err = abs(obs_hz - rate) / rate
                    if rel_err < tol_rate * 1.5 and rel_err < best["err"]:
                        best = {
                            "ch": ch,
                            "num_ch": "est",
                            "blocks": blocks_int,
                            "obs_hz": obs_hz,
                            "err": rel_err,
                            "note": "timing_fallback",
                        }

    # --- 3. Combined-group fallback ---
    if best["ch"] == "uncertain":
        combined_groups = [("CH256", "CH64"), ("CH256", "CH52")]
        for g1, g2 in combined_groups:
            rate1, rate2 = EXPECTED_RATES[g1], EXPECTED_RATES[g2]
            combined_rate = rate1 + rate2
            num_ch1, num_ch2 = max(EXPECTED_GROUPS[g1]), max(EXPECTED_GROUPS[g2])
            bps_avg = (BITS_PER_SAMPLE[g1] + BITS_PER_SAMPLE[g2]) / 2
            data_bytes = max(
                median_len - (overhead_candidates[0] if overhead_candidates else 3), 0
            )
            median_bits = data_bytes * 8
            if median_bits <= 0:
                continue
            spp_size = median_bits / ((num_ch1 + num_ch2) * bps_avg)
            if spp_size <= 0:
                continue
            obs_hz = pkt_hz * spp_size
            rel_err = abs(obs_hz - combined_rate) / combined_rate
            if rel_err < tol_rate * 1.5 and rel_err < best["err"]:
                best = {
                    "ch": f"{g1}+{g2}",
                    "num_ch": num_ch1 + num_ch2,
                    "blocks": None,
                    "obs_hz": obs_hz,
                    "err": rel_err,
                    "note": "combined_fallback",
                }

    # Attach notes
    if best:
        notes = []
        if note_tags:
            notes.extend(note_tags)
        notes.extend(extra_note)
        if notes:
            best["note"] = (
                f"{best.get('note', '')};" + ";".join(notes)
                if best.get("note")
                else ";".join(notes)
            )

    return best


# File analysis function ============================================================
def analyze_file(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    times, uuids, data = parse_lines(lines)

    # Sequence-level headers (arrival order) if needed elsewhere
    ordered_headers = [p[0] for p in data if p]
    period, purity, profiles = detect_header_cycle(ordered_headers, max_period=16)

    # --- derive per-header sequence purity ---
    # Try sequence_groups first (expects profiles from detect_header_cycle)
    per_hdr_purity = {}
    if profiles:
        # sequence_groups expects profiles and returns hdr -> {'positions': [...], 'support': x}
        try:
            seq_map = sequence_groups(profiles, period, min_share=0.01)
        except Exception:
            seq_map = {}

        if seq_map:
            # keys in seq_map are header bytes (ints)
            for h, entry in seq_map.items():
                per_hdr_purity[h] = float(entry.get("support", 0.0))
        else:
            # fallback: compute purity from raw profiles: top-position fraction
            import collections

            hdr_counts = collections.defaultdict(int)
            hdr_top = collections.defaultdict(int)
            for pos_prof in profiles:
                for h, c in pos_prof.items():
                    hdr_counts[h] += c
                    if c > hdr_top[h]:
                        hdr_top[h] = c
            for h in hdr_counts:
                per_hdr_purity[h] = (
                    float(hdr_top[h]) / float(hdr_counts[h]) if hdr_counts[h] else 0.0
                )
    # if no profiles, per_hdr_purity remains empty and we default to 0.0 below

    by_hdr = defaultdict(lambda: {"times": [], "payloads": []})
    for t, p in zip(times, data):
        if not p:
            continue
        by_hdr[p[0]]["times"].append(t)
        by_hdr[p[0]]["payloads"].append(p)

    # small helper for safe float formatting
    import math

    def fmt_opt(x, fmt="{:.3f}"):
        return (
            fmt.format(x)
            if (x is not None and not (isinstance(x, float) and math.isnan(x)))
            else "None"
        )

    rows = []
    for hdr, info in by_hdr.items():
        # Raw per-header packet rate
        pkt_hz_raw = packet_frequency(info["times"]) or 0.0

        # Sequence-derived hint (if cycle strong)
        seq_hint = None
        if (period is not None) and (purity is not None) and (purity >= 0.7):
            seq_hint = sequence_rate_hint(info["times"], period, ordered_headers, hdr)

        # Blend for stability when we have a hint
        pkt_hz_blend = pkt_hz_raw
        if seq_hint is not None and seq_hint > 0:
            pkt_hz_blend = 0.6 * pkt_hz_raw + 0.4 * seq_hint

        # Pass estimator and blended rate to classifier
        res = classify_header(
            info["times"],
            info["payloads"],
            tol_blocks=0.10,
            tol_rate=0.15,
            overhead_fn=estimate_overhead_bytes,
            note_tags=[
                f"hdr=0x{hdr:02x}",
                f"period={period}" if period is not None else "period=None",
                f"purity={purity:.3f}" if purity is not None else "purity=None",
                f"seq_hint={fmt_opt(seq_hint)}",
                f"pkt_hz_raw={pkt_hz_raw:.3f}",
                f"pkt_hz_blend={pkt_hz_blend:.3f}",
            ],
            pkt_hz_hint=pkt_hz_blend,  # <= use blended rate
        )

        row = {
            "Header": f"0x{hdr:02x}",
            "HdrByte": hdr,  # raw integer header for joins/diagnostics
            "HighNibble": (hdr >> 4) & 0xF,
            "LowNibble": hdr & 0xF,
            "Packets": len(info["times"]),
            "PktHzRaw": pkt_hz_raw,
            "PktHzSeqHint": seq_hint if seq_hint is not None else np.nan,
            "PktHzBlend": pkt_hz_blend,
            "MedianLen": np.median([len(p) for p in info["payloads"]]),
            "BestCH": None,
            "BlocksPerPkt": None,
            "ObsHz": None,
            "RelErr": None,
            "SequencePurity": per_hdr_purity.get(hdr, 0.0),
            "Note": None,
        }
        if res:
            row.update(
                {
                    "BestCH": res.get("ch"),
                    "BlocksPerPkt": res.get("blocks"),
                    "ObsHz": res.get("obs_hz"),
                    "RelErr": res.get("err"),
                    "Note": res.get("note"),
                }
            )
        rows.append(row)

    return pd.DataFrame(rows)


# Diagnostic functions ================================================================
def build_decoder_summary(df_all: pd.DataFrame) -> pd.DataFrame:
    # Keep only rows with some classification
    df_valid = df_all.dropna(subset=["BestCH"])

    # Aggregate
    summary = (
        df_valid.groupby("BestCH")
        .agg(
            ExpectedHz=("ObsHz", "median"),
            MedianLen=("MedianLen", "median"),
            BlocksPerPkt=("BlocksPerPkt", "median"),
            TypicalHeaders=("Header", lambda x: ", ".join(sorted(set(x)))),
            RelErr=("RelErr", "median"),
        )
        .reset_index()
    )

    # Round numbers for readability
    summary["ExpectedHz"] = summary["ExpectedHz"].round(1)
    summary["MedianLen"] = summary["MedianLen"].round(0).astype("Int64")
    summary["BlocksPerPkt"] = summary["BlocksPerPkt"].round(0).astype("Int64")
    summary["RelErr"] = summary["RelErr"].round(3)

    return summary


def build_header_summary(combined: pd.DataFrame) -> pd.DataFrame:
    """
    Build a summary table with headers as rows and filenames as columns,
    showing the most likely channel type (BestCH) if the header appears
    in that file, otherwise an empty string.
    """
    # For each (Header, File), pick the most common BestCH
    type_map = (
        combined.groupby(["Header", "File"])["BestCH"]
        .agg(lambda x: x.mode()[0] if not x.mode().empty else "uncertain")
        .unstack(fill_value="")
    )
    return type_map


def plot_header_histograms(
    dfs: Sequence[pd.DataFrame],
    *,
    file_col: str = "File",
    header_col: str = "Header",
    count_col: str = "Packets",
    figsize=(4, 3),
    cmap_name="tab10",
):
    pairs = []
    for df in dfs:
        if file_col not in df.columns:
            raise ValueError(f"Missing column '{file_col}'")
        fname = df[file_col].iloc[0]
        pairs.append((fname, df.copy()))

    if not pairs:
        raise ValueError("No dataframes provided.")

    all_headers = sorted(
        {h for _, df in pairs for h in df[header_col].dropna().unique()},
        key=lambda s: int(s, 16) if isinstance(s, str) and s.startswith("0x") else s,
    )

    n_files = len(pairs)
    ncols = 3
    nrows = int(np.ceil(n_files / ncols))
    fig_width = figsize[0] * ncols
    fig_height = figsize[1] * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
    axes = axes.flatten()

    cmap = cm.get_cmap(cmap_name, n_files)

    for i, (fname, df) in enumerate(pairs):
        ax = axes[i]
        counts = {h: 0 for h in all_headers}
        for _, row in df.iterrows():
            h = row[header_col]
            c = int(row.get(count_col, 1) or 0)
            if h in counts:
                counts[h] += c
        total = sum(counts.values()) or 1
        proportions = [counts[h] / total for h in all_headers]

        x = np.arange(len(all_headers))
        ax.bar(x, proportions, color=cmap(i), edgecolor="k", alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(all_headers, rotation=90, fontsize=8)
        ax.set_ylim(0, max(0.1, max(proportions) * 1.15))
        ax.set_title(Path(fname).name, fontsize=10)
        ax.set_ylabel("Proportion")

        for xi, p in zip(x, proportions):
            if p > 0.02:
                ax.text(xi, p + 0.005, f"{p:.2f}", ha="center", va="bottom", fontsize=7)

    # Hide unused subplots
    for j in range(n_files, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    return fig


# Main execution ==================================================================
if __name__ == "__main__":
    all_results = []
    for fname in os.listdir("data_raw"):
        if not fname.endswith(".txt"):
            continue
        df = analyze_file(os.path.join("data_raw", fname))
        df["File"] = fname
        print(f"\n=== {fname} ===")
        print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
        all_results.append(df)
    # consistency across files
    if all_results:
        combined = pd.concat(all_results)
        summary = (
            combined.groupby("Header")
            .agg(
                FileCount=("File", "count"),
                BestGuessCH=(
                    "BestCH",
                    lambda x: x.mode()[0] if not x.mode().empty else "uncertain",
                ),
                BlocksPerPkt_Median=("BlocksPerPkt", "median"),
                BlocksPerPkt_SD=("BlocksPerPkt", "std"),
                ObsHz_Median=("ObsHz", "median"),
                ObsHz_SD=("ObsHz", "std"),
                HighNibble_Mode=(
                    "HighNibble",
                    lambda s: int(s.mode().iloc[0]) if not s.mode().empty else np.nan,
                ),
                LowNibble_Mode=(
                    "LowNibble",
                    lambda s: int(s.mode().iloc[0]) if not s.mode().empty else np.nan,
                ),
                HighNibble_UniqueCount=(
                    "HighNibble",
                    lambda s: int(s.dropna().nunique()),
                ),
                RelErr_Median=("RelErr", "median"),
                SequencePurity_Median=("SequencePurity", "median"),
            )
            .sort_values(by="ObsHz_Median", ascending=False)
        )
        print("\n=== Header Summary ===")
        print(summary.to_markdown(floatfmt=(".2f")))

        # Notes:
        # - HighNibble_Mode quickly validates the repo assertion that high nibble encodes frequency (expect modes 0x1,0x3,0x4 for 256/64/52 Hz).
        # - LowNibble_Mode shows the dominant data-type encoding (EEG/OPTICS/IMU variants).
        # - HighNibble_UniqueCount > 1 indicates potential capture issues, spoofed packets, multiplexing, or file-to-file differences — these should be triaged first.
        # - Use HighNibble_Mode together with ObsHz_Median to detect misclassifications: if HighNibble_Mode suggests 0x1 (256Hz) but ObsHz_Median ≈ 64, you likely have mixed payloads or wrong overhead assumptions.

        # Summary
        summary_table = build_decoder_summary(combined)
        print(summary_table.to_markdown(floatfmt=(".2f")))
        print(summary_table.to_string(index=False))

        presence_matrix = build_header_summary(combined)
        print("\n=== Header presence across presets ===")
        print(presence_matrix.to_string())

        # Plot
        fig = plot_header_histograms(all_results)
        fig.show()
        fig.savefig("header_histograms.png", dpi=150)

