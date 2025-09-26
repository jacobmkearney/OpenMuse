#!/usr/bin/env python3
"""
Validate Muse S (Athena) packet sequence continuity and compute data loss per file.

Assumptions:
- Each line in the input files is tab-separated: "<host_iso_timestamp>\t...\t<payload_hex>"
- The 1-byte sequence number is at offset 1 within the payload (0-based).
- Device uses a 1-byte counter that increments by 1 and wraps 255 -> 0 (256 states total).
- Wrap-around is handled by modular arithmetic; missing packets are inferred from modular deltas > 1.

Outputs:
- A per-file summary table with packet counts, inferred lost packets, loss rate, etc.
- (Optional) verbose details for files that show loss/out-of-order/duplicates.

Usage:
- Place this script next to your `data_raw/` directory or adjust DATA_DIR.
- Run: python validate_seq_loss.py
"""

import os
import struct
import sys
from datetime import datetime

import pandas as pd

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATA_DIR = "./data_raw/"
VERBOSE = False  # Set True to print per-file detailed diagnostics
ORDER_BY_DEVICE_TIME = False  # If True and device time is available, sort by it before analysis


# -----------------------------------------------------------------------------
# Parsing helpers
# -----------------------------------------------------------------------------
def parse_line(line: str):
    """
    Expect at least 3 tab-separated fields where the 3rd is the hex payload.
    Returns:
        host_ts (float seconds since epoch),
        payload (bytes) or (None, None) if parsing fails.
    """
    parts = line.strip().split("\t")
    if len(parts) < 3:
        return None, None
    try:
        host_ts = datetime.fromisoformat(parts[0].replace("Z", "+00:00")).timestamp()
    except Exception:
        host_ts = None
    try:
        payload = bytes.fromhex(parts[2])
    except Exception:
        return host_ts, None
    return host_ts, payload


def extract_seq_and_time(payload: bytes):
    """
    Extract sequence number (offset 1) and device time (uint32 LE at offset 2) if present.
    Returns (seq:int, dev_t_s:float|None).
    """
    if payload is None or len(payload) < 2:
        return None, None
    seq = payload[1]
    dev_t_s = None
    if len(payload) >= 6:  # device time available as 4B LE at offset 2
        try:
            dev_ms = struct.unpack_from("<I", payload, 2)[0]
            dev_t_s = dev_ms * 1e-3
        except Exception:
            dev_t_s = None
    return seq, dev_t_s


# -----------------------------------------------------------------------------
# Core analysis
# -----------------------------------------------------------------------------
def analyze_sequences(records):
    """
    records: list of dicts with keys: {"seq", "host_t_s", "dev_t_s"}

    Returns a summary dict with:
        n_packets, unique_seq_values, min_seq, max_seq,
        wraps, out_of_order_events, duplicates,
        lost_packets, loss_rate, consecutive_ratio
    """
    if not records:
        return {
            "n_packets": 0,
            "unique_seq_values": 0,
            "min_seq": None,
            "max_seq": None,
            "wraps": 0,
            "out_of_order_events": 0,
            "duplicates": 0,
            "lost_packets": 0,
            "loss_rate": 0.0,
            "consecutive_ratio": None,
            "saw_zero": False,
            "saw_255": False,
        }

    # Choose ordering
    if ORDER_BY_DEVICE_TIME and all(r["dev_t_s"] is not None for r in records):
        records = sorted(records, key=lambda r: (r["dev_t_s"]))
    else:
        # Preserve file order (arrival order)
        pass

    seqs = [r["seq"] for r in records if r["seq"] is not None]

    n_packets = len(seqs)
    if n_packets < 2:
        return {
            "n_packets": n_packets,
            "unique_seq_values": len(set(seqs)),
            "min_seq": min(seqs) if seqs else None,
            "max_seq": max(seqs) if seqs else None,
            "wraps": 0,
            "out_of_order_events": 0,
            "duplicates": 0,
            "lost_packets": 0,
            "loss_rate": 0.0,
            "consecutive_ratio": None,
            "saw_zero": (0 in seqs),
            "saw_255": (255 in seqs),
        }

    wraps = 0
    out_of_order = 0
    duplicates = 0
    lost_packets = 0
    consecutive_pairs = 0

    prev = seqs[0]
    for curr in seqs[1:]:
        # Raw delta for diagnostics; modular delta for logic
        raw_delta = curr - prev
        delta_mod = (curr - prev) & 0xFF  # 0..255

        if delta_mod == 1:
            consecutive_pairs += 1
        elif delta_mod == 0:
            # exact duplicate (retransmit or repeated frame id)
            duplicates += 1
        else:
            # Count missing frames between prev and curr
            lost_packets += delta_mod - 1

        # Wrap detection heuristic: 255 -> 0 is the "clean" wrap
        if prev == 255 and curr == 0:
            wraps += 1
        # Out-of-order hint: negative raw delta that isn't the clean wrap case
        if raw_delta < 0 and not (prev == 255 and curr == 0):
            out_of_order += 1

        prev = curr

    # Loss rate = lost / (received + lost)
    denom = n_packets + lost_packets
    loss_rate = (lost_packets / denom) if denom > 0 else 0.0
    consecutive_ratio = consecutive_pairs / (n_packets - 1) if n_packets > 1 else None

    return {
        "n_packets": n_packets,
        "unique_seq_values": len(set(seqs)),
        "min_seq": min(seqs),
        "max_seq": max(seqs),
        "wraps": wraps,
        "out_of_order_events": out_of_order,
        "duplicates": duplicates,
        "lost_packets": lost_packets,
        "loss_rate": loss_rate,
        "consecutive_ratio": consecutive_ratio,
        "saw_zero": (0 in seqs),
        "saw_255": (255 in seqs),
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def validate_directory(data_dir: str):
    files = sorted(p for p in os.listdir(data_dir) if p.endswith(".txt"))
    all_summaries = []

    for fn in files:
        path = os.path.join(data_dir, fn)
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                host_ts, payload = parse_line(line)
                if payload is None:
                    continue
                seq, dev_t_s = extract_seq_and_time(payload)
                if seq is None:
                    continue
                # Range validation (byte) will be implicit by seq extraction
                records.append({"seq": seq, "host_t_s": host_ts, "dev_t_s": dev_t_s})

        summary = analyze_sequences(records)
        # Additional boolean validations for readability
        in_range = all(0 <= r["seq"] <= 255 for r in records) if records else True
        summary.update(
            {
                "file": fn,
                "values_in_0_255": in_range,
                # Optional: show whether both extremes were seen at least once
                "covers_both_extremes": (summary["saw_zero"] and summary["saw_255"]),
            }
        )
        all_summaries.append(summary)

        if VERBOSE:
            print(f"\n--- {fn} ---")
            for k, v in summary.items():
                if k == "file":
                    continue
                print(f"{k:24s}: {v}")

    if not all_summaries:
        print("No .txt files found or no valid packets parsed.")
        return

    df = pd.DataFrame(
        all_summaries,
        columns=[
            "file",
            "n_packets",
            "unique_seq_values",
            "min_seq",
            "max_seq",
            "values_in_0_255",
            "covers_both_extremes",
            "wraps",
            "out_of_order_events",
            "duplicates",
            "lost_packets",
            "loss_rate",
            "consecutive_ratio",
        ],
    )

    print("\nPacket continuity & loss summary:")
    # Pretty print with percentages
    out = df.copy()
    out["loss_rate"] = (out["loss_rate"] * 100).map("{:.3f}%".format)
    out["consecutive_ratio"] = (out["consecutive_ratio"] * 100).map("{:.3f}%".format)
    print(out.to_markdown(index=False))


if __name__ == "__main__":
    # Allow optional args: [DATA_DIR] [--device-order] [--verbose]
    args = sys.argv[1:]
    global DATA_DIR, ORDER_BY_DEVICE_TIME, VERBOSE
    if args:
        for a in args:
            if a == "--device-order":
                ORDER_BY_DEVICE_TIME = True
            elif a == "--verbose":
                VERBOSE = True
            elif os.path.isdir(a):
                DATA_DIR = a
            else:
                print(f"Unrecognized argument: {a}", file=sys.stderr)

    validate_directory(DATA_DIR)
# Packet continuity & loss summary:
# | file           |   n_packets |   unique_seq_values |   min_seq |   max_seq | values_in_0_255   | covers_both_extremes   |   wraps |   out_of_order_events |   duplicates |   lost_packets | loss_rate   | consecutive_ratio   |
# |:---------------|------------:|--------------------:|----------:|----------:|:------------------|:-----------------------|--------:|----------------------:|-------------:|---------------:|:------------|:--------------------|
# | data_p1034.txt |        1245 |                 256 |         0 |       255 | True              | True                   |       4 |                     0 |            0 |              0 | 0.000%      | 100.000%            |
# | data_p1035.txt |        1036 |                 256 |         0 |       255 | True              | True                   |       4 |                     0 |            0 |              0 | 0.000%      | 100.000%            |
# | data_p1041.txt |        2195 |                 256 |         0 |       255 | True              | True                   |       8 |                     0 |            0 |              0 | 0.000%      | 100.000%            |
# | data_p1042.txt |        2194 |                 256 |         0 |       255 | True              | True                   |       8 |                     0 |            0 |              0 | 0.000%      | 100.000%            |
# | data_p1043.txt |        1810 |                 256 |         0 |       255 | True              | True                   |       7 |                     0 |            0 |              0 | 0.000%      | 100.000%            |
# | data_p1044.txt |        1807 |                 256 |         0 |       255 | True              | True                   |       7 |                     0 |            0 |              0 | 0.000%      | 100.000%            |
# | data_p1045.txt |        1589 |                 256 |         0 |       255 | True              | True                   |       6 |                     0 |            0 |              0 | 0.000%      | 100.000%            |
# | data_p1046.txt |        1596 |                 256 |         0 |       255 | True              | True                   |       6 |                     0 |            0 |              0 | 0.000%      | 100.000%            |
# | data_p20.txt   |         837 |                 256 |         0 |       255 | True              | True                   |       3 |                     0 |            0 |              0 | 0.000%      | 100.000%            |
# | data_p21.txt   |         837 |                 256 |         0 |       255 | True              | True                   |       3 |                     0 |            0 |              0 | 0.000%      | 100.000%            |
# | data_p4129.txt |        1597 |                 256 |         0 |       255 | True              | True                   |       6 |                     0 |            0 |              0 | 0.000%      | 100.000%            |
# | data_p50.txt   |         836 |                 256 |         0 |       255 | True              | True                   |       3 |                     0 |            0 |              0 | 0.000%      | 100.000%            |
# | data_p51.txt   |         836 |                 256 |         0 |       255 | True              | True                   |       3 |                     0 |            0 |              0 | 0.000%      | 100.000%            |
# | data_p60.txt   |         836 |                 256 |         0 |       255 | True              | True                   |       3 |                     0 |            0 |              0 | 0.000%      | 100.000%            |
# | data_p61.txt   |         837 |                 256 |         0 |       255 | True              | True                   |       3 |                     0 |            0 |              0 | 0.000%      | 100.000%            |
