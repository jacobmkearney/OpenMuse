"""
infodata_inmemory.py

Scan all .txt files under DATA_DIR, extract SUBPKT_UNKNOWN2 (bytes 10..13),
interpret the 4 bytes in several ways, compute diagnostics and heuristics,
and print a compact markdown summary - all in memory, no disk writes.

Expect line format:
  <ISO8601 timestamp>\\t<uuid>\\t<hex payload>

Usage:
  python infodata_inmemory.py
"""

import os
import struct
import math
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd

# ----------------------------- Config ------------------------------------
DATA_DIR = "./data_raw/"
FILES = sorted(p for p in os.listdir(DATA_DIR) if p.endswith(".txt"))

FREQ_MAP = {
    1: 256.0,
    2: 128.0,
    3: 64.0,
    4: 52.0,
    5: 32.0,
    6: 16.0,
    7: 10.0,
    8: 1.0,
    9: 0.1,
}
TYPE_MAP = {
    1: "EEG4",
    2: "EEG8",
    3: "REF",
    4: "Optics4",
    5: "Optics8",
    6: "Optics16",
    7: "ACCGYRO",
    8: "Battery",
}

TOP_K = 6  # how many top frequencies to show in diagnostics


# ----------------------------- Utilities --------------------------------
def parse_line(line: str):
    parts = line.strip().split("\t")
    if len(parts) < 3:
        return None
    try:
        ts = datetime.fromisoformat(parts[0].replace("Z", "+00:00")).timestamp()
    except Exception:
        ts = None
    try:
        payload = bytes.fromhex(parts[2])
    except Exception:
        payload = None
    return ts, payload


def extract_pkt_type(payload: bytes) -> Tuple[Any, Any]:
    if not payload or len(payload) <= 9:
        return None, None
    id_byte = payload[9]
    freq_code = (id_byte >> 4) & 0x0F
    type_code = id_byte & 0x0F
    return FREQ_MAP.get(freq_code), TYPE_MAP.get(type_code, f"TYPE_{type_code}")


def entropy_from_counts(counts: Dict) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    e = 0.0
    for v in counts.values():
        p = v / total
        if p > 0:
            e -= p * math.log(p)
    return e


def longest_run(arr: List) -> int:
    if not arr:
        return 0
    max_run = 1
    run = 1
    for i in range(1, len(arr)):
        if arr[i] == arr[i - 1]:
            run += 1
            if run > max_run:
                max_run = run
        else:
            run = 1
    return max_run


# ------------------------ 4-byte interpretations -------------------------
def interpret_4bytes(b4: bytes) -> Dict[str, object]:
    assert len(b4) == 4
    out = {}
    out["hex_le"] = b4.hex()
    out["u32_le"] = struct.unpack("<I", b4)[0]
    out["i32_le"] = struct.unpack("<i", b4)[0]
    out["f32_le"] = struct.unpack("<f", b4)[0]
    out["u16_0_le"] = struct.unpack("<H", b4[0:2])[0]
    out["u16_1_le"] = struct.unpack("<H", b4[2:4])[0]
    out["u8_0"] = b4[0]
    out["u8_1"] = b4[1]
    out["u8_2"] = b4[2]
    out["u8_3"] = b4[3]
    out["u32_be"] = struct.unpack(">I", b4)[0]
    out["i32_be"] = struct.unpack(">i", b4)[0]
    out["f32_be"] = struct.unpack(">f", b4)[0]
    return out


# ----------------------- summarisation helpers ---------------------------
def summarise_numeric(arr: np.ndarray) -> Dict:
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
    }


def summarise_sequence_of_vals(values: List):
    res = {}
    n = len(values)
    res["n"] = n
    if n == 0:
        return res
    arr = np.array(values, dtype=np.float64)
    res.update(summarise_numeric(arr))
    ctr = Counter(values)
    res["n_unique"] = len(ctr)
    topk = ctr.most_common(TOP_K)
    res["topk"] = [(str(k), int(v), float(v / n)) for k, v in topk]
    res["entropy"] = float(entropy_from_counts(ctr))
    if n >= 2:
        diffs = np.diff(arr)
        res["diff_median"] = float(np.median(diffs))
        res["diff_mean"] = float(np.mean(diffs))
        res["diff_std"] = float(np.std(diffs))
        res["prop_diffs_eq_1"] = float(np.mean(diffs == 1))
        mode_diff = (
            Counter(diffs.tolist()).most_common(1)[0][0] if diffs.size > 0 else 0
        )
        res["mode_diff"] = float(mode_diff)
        res["prop_diffs_eq_mode"] = float(np.mean(diffs == mode_diff))
        res["prop_diffs_zero"] = float(np.mean(diffs == 0))
    else:
        res.update(
            {
                "diff_median": None,
                "diff_mean": None,
                "diff_std": None,
                "prop_diffs_eq_1": None,
                "mode_diff": None,
                "prop_diffs_eq_mode": None,
                "prop_diffs_zero": None,
            }
        )
    res["longest_run"] = longest_run(values)
    return res


# ------------------------- core processing --------------------------------
def process_files_in_memory(files: List[str], data_dir: str = DATA_DIR):
    per_type_rows = defaultdict(list)

    for fn in files:
        path = os.path.join(data_dir, fn)
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as fh:
            for ln in fh:
                parsed = parse_line(ln)
                if not parsed:
                    continue
                ts, payload = parsed
                if not payload or len(payload) < 14:
                    continue
                _, pkt_type = extract_pkt_type(payload)
                b4 = payload[10:14]
                if len(b4) != 4:
                    continue
                interp = interpret_4bytes(b4)
                row = {
                    "file": fn,
                    "ts": ts,
                    "pkt_type": pkt_type,
                    "raw_hex": interp["hex_le"],
                    "u32_le": int(interp["u32_le"]),
                    "i32_le": int(interp["i32_le"]),
                    "f32_le": float(interp["f32_le"]),
                    "u16_0_le": int(interp["u16_0_le"]),
                    "u16_1_le": int(interp["u16_1_le"]),
                    "u8_0": int(interp["u8_0"]),
                    "u8_1": int(interp["u8_1"]),
                    "u8_2": int(interp["u8_2"]),
                    "u8_3": int(interp["u8_3"]),
                    "u32_be": int(interp["u32_be"]),
                }
                per_type_rows[pkt_type].append(row)

    return per_type_rows


def aggregate_in_memory(per_type_rows: Dict[str, List[Dict]]):
    summaries = {}
    for pkt_type, rows in per_type_rows.items():
        summaries[pkt_type] = {}
        if not rows:
            continue
        # For each interpretation, gather sequence and summarise
        keys_to_check = [
            "u32_le",
            "i32_le",
            "f32_le",
            "u16_0_le",
            "u16_1_le",
            "u8_0",
            "u8_1",
            "u8_2",
            "u8_3",
            "u32_be",
        ]
        for key in keys_to_check:
            seq = [r[key] for r in rows if r.get(key) is not None]
            s = summarise_sequence_of_vals(seq)
            # heuristic counter-like test
            counter_like = False
            if s.get("n", 0) >= 3:
                n_unique = s.get("n_unique", 0)
                diffs_ok = (s.get("prop_diffs_eq_1") or 0.0) > 0.4
                if (n_unique > max(10, 0.5 * s["n"])) and diffs_ok:
                    counter_like = True
            s["counter_like"] = counter_like
            summaries[pkt_type][key] = s
    return summaries


# ----------------------------- compact table --------------------------------
def heuristic_label(s: Dict) -> str:
    if s.get("counter_like"):
        return "counter"
    n = s.get("n") or 0
    n_unique = s.get("n_unique") or 0
    longest = s.get("longest_run") or 0
    entropy = s.get("entropy") or 0.0
    prop1 = s.get("prop_diffs_eq_1") or 0.0
    if n > 0 and n_unique <= 3 and (longest >= 0.5 * n):
        return "flag/constant"
    if entropy < 0.3 and n_unique <= max(3, 0.02 * max(1, n)):
        return "constant"
    if prop1 > 0.4:
        return "counter-like (step 1)"
    return "unknown"


def build_compact_df(summaries: Dict[str, Dict]) -> pd.DataFrame:
    rows = []
    for pkt_type, interps in summaries.items():
        for interp_name, s in interps.items():
            rows.append(
                {
                    "pkt_type": pkt_type,
                    "interpretation": interp_name,
                    "n": s.get("n"),
                    "n_unique": s.get("n_unique"),
                    "min": s.get("min"),
                    "max": s.get("max"),
                    "mean": s.get("mean"),
                    "std": s.get("std"),
                    "entropy": s.get("entropy"),
                    "diff_median": s.get("diff_median"),
                    "diff_std": s.get("diff_std"),
                    "prop_diffs_eq_1": s.get("prop_diffs_eq_1"),
                    "prop_diffs_eq_mode": s.get("prop_diffs_eq_mode"),
                    "prop_diffs_zero": s.get("prop_diffs_zero"),
                    "longest_run": s.get("longest_run"),
                    "counter_like": bool(s.get("counter_like", False)),
                    "topk": "; ".join(
                        [f"{t[0]} ({t[1]}/{t[2]:.2f})" for t in (s.get("topk") or [])]
                    ),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["heuristic_guess"] = df.apply(lambda r: heuristic_label(r.to_dict()), axis=1)
    # round numeric columns for readability
    for c in [
        "min",
        "max",
        "mean",
        "std",
        "entropy",
        "diff_median",
        "diff_std",
        "prop_diffs_eq_1",
        "prop_diffs_eq_mode",
        "prop_diffs_zero",
    ]:
        if c in df.columns:
            df[c] = df[c].apply(
                lambda v: (
                    round(v, 4)
                    if (
                        isinstance(v, (int, float))
                        and not (math.isnan(v) or math.isinf(v))
                    )
                    else v
                )
            )
    # reorder columns for readability
    cols = [
        "pkt_type",
        "interpretation",
        "heuristic_guess",
        "n",
        "n_unique",
        "min",
        "max",
        "mean",
        "std",
        "entropy",
        "diff_median",
        "diff_std",
        "prop_diffs_eq_1",
        "prop_diffs_eq_mode",
        "prop_diffs_zero",
        "longest_run",
        "counter_like",
        "topk",
    ]
    available_cols = [c for c in cols if c in df.columns]
    return df[available_cols]


# ------------------------------- main -------------------------------------
def main(files: List[str] = FILES, data_dir: str = DATA_DIR) -> pd.DataFrame:
    if len(files) == 0:
        raise SystemExit(f"No .txt files found in {data_dir}")
    per_type_rows = process_files_in_memory(files, data_dir=data_dir)
    if not per_type_rows:
        raise SystemExit("No valid packets found - check DATA_DIR and file format.")
    summaries = aggregate_in_memory(per_type_rows)
    compact_df = build_compact_df(summaries)
    # Print compact markdown table - truncated topk for readability
    if compact_df.empty:
        print("No interpretable values found.")
    else:
        disp = compact_df.copy()
        disp["topk"] = disp["topk"].apply(
            lambda s: (s[:120] + "...") if isinstance(s, str) and len(s) > 120 else s
        )
        print(
            "\nCompact INFODATA summary (in-memory) - one row per pkt_type / interpretation\n"
        )
        print(disp.to_markdown(index=False))
    return compact_df


if __name__ == "__main__":
    df_result = main()


# ========================================================================
# TEST ===================================================================
# ========================================================================
# infodata_tag_tests.py
import os, struct
from collections import Counter, defaultdict
from datetime import datetime
import numpy as np
import pandas as pd

DATA_DIR = "./data_raw/"
FILES = sorted(p for p in os.listdir(DATA_DIR) if p.endswith(".txt"))
TYPE_MAP = {
    1: "EEG4",
    2: "EEG8",
    3: "REF",
    4: "Optics4",
    5: "Optics8",
    6: "Optics16",
    7: "ACCGYRO",
    8: "Battery",
}


def parse_line(line):
    parts = line.strip().split("\t")
    if len(parts) < 3:
        return None, None
    try:
        ts = datetime.fromisoformat(parts[0].replace("Z", "+00:00")).timestamp()
    except Exception:
        ts = None
    try:
        payload = bytes.fromhex(parts[2])
    except Exception:
        payload = None
    return ts, payload


def get_pkt_type(payload):
    if not payload or len(payload) <= 9:
        return None
    idb = payload[9]
    type_code = idb & 0x0F
    return TYPE_MAP.get(type_code, f"TYPE_{type_code}")


# helper to find all occurrences of a single-byte tag value
def find_all(b: bytes, tag_byte: int):
    pos = []
    start = 0
    while True:
        i = b.find(bytes([tag_byte]), start)
        if i == -1:
            break
        pos.append(i)
        start = i + 1
    return pos


records = []
for fn in FILES:
    path = os.path.join(DATA_DIR, fn)
    if not os.path.isfile(path):
        continue
    with open(path, "r", encoding="utf-8") as fh:
        for ln in fh:
            ts, payload = parse_line(ln)
            if not payload or len(payload) < 14:
                continue
            pkt_type = get_pkt_type(payload)
            # extract unknown2
            b10_13 = payload[10:14]
            if len(b10_13) != 4:
                continue
            u16_0 = struct.unpack_from("<H", b10_13, 0)[0]
            u16_1 = struct.unpack_from("<H", b10_13, 2)[0]
            # compute data payload (after header 14)
            data = payload[14:]
            data_len = len(data)
            # find tag occurrences (0x47) - adjust if you want different tag
            tag_positions = find_all(data, 0x47)
            tag_count = len(tag_positions)
            # distances between tag starts
            tag_dists = (
                [y - x for x, y in zip(tag_positions[:-1], tag_positions[1:])]
                if tag_count >= 2
                else []
            )
            first_tag_pos = tag_positions[0] if tag_count >= 1 else None
            records.append(
                {
                    "file": fn,
                    "pkt_type": pkt_type,
                    "u16_0": u16_0,
                    "u16_1": u16_1,
                    "data_len": data_len,
                    "tag_count": tag_count,
                    "tag_positions": tag_positions,
                    "first_tag_pos": first_tag_pos,
                    "tag_dists": tag_dists,
                    "payload_hex_sample": payload.hex()[:200],
                }
            )

df = pd.DataFrame(records)
if df.empty:
    raise SystemExit("No packets found.")

# 1) How often does u16_1 equal tag_count?
print("\n=== u16_1 == tag_count proportion by packet type ===")
for t, g in df.groupby("pkt_type"):
    prop = (g["u16_1"] == g["tag_count"]).mean()
    print(
        f"{t}: n={len(g)}, prop(u16_1==tag_count) = {prop:.3f}, mean_tag_count={g['tag_count'].mean():.2f}"
    )

# 2) Summarise tag_count distribution vs u16_1
print("\n=== tag_count distribution grouped by u16_1 (per pkt_type) ===")
for t, g in df.groupby("pkt_type"):
    print(f"\n-- {t} (n={len(g)}) --")
    pivot = g.groupby("u16_1")["tag_count"].describe()[["count", "mean", "min", "max"]]
    print(pivot)

# 3) Examine typical tag distance (mode) by pkt_type and u16_1
print("\n=== most common tag distance (mode) per pkt_type / u16_1 ===")
for t, g in df.groupby("pkt_type"):
    inner = {}
    for u, sub in g.groupby("u16_1"):
        # flatten tag_dists lists
        dists = []
        for d in sub["tag_dists"]:
            dists.extend(d if isinstance(d, list) else [])
        if not dists:
            mode = None
        else:
            mode = Counter(dists).most_common(1)[0][0]
        inner[u] = {
            "n_packets": len(sub),
            "mean_tag_count": sub["tag_count"].mean(),
            "mode_dist": mode,
        }
    print(f"{t}: {inner}")

# 4) Correlate u16_1 with first_tag_pos (does u16_1 indicate start offset?)
print("\n=== correlation u16_1 vs first_tag_pos (per type) ===")
for t, g in df.groupby("pkt_type"):
    # remove None
    sub = g[g["first_tag_pos"].notnull()]
    if len(sub) < 10:
        continue
    corr = np.corrcoef(
        sub["u16_1"].astype(float).values, sub["first_tag_pos"].astype(float).values
    )[0, 1]
    print(f"{t}: n={len(sub)}, corr(u16_1, first_tag_pos) = {corr:.3f}")

# 5) Print a few sample payloads per pkt_type/ u16_1 for manual inspection
print("\n=== sample payload hex (first 3 per pkt_type/u16_1) ===")
for t, g in df.groupby("pkt_type"):
    print(f"\n-- {t} --")
    for u, sub in g.groupby("u16_1"):
        sample_hexes = sub["payload_hex_sample"].unique()[:3].tolist()
        print(f"u16_1={u}: n={len(sub)}, sample hexes: {sample_hexes}")

print("\nDone.")
