import argparse
import csv
import os
from collections import Counter


def load_segmented(csv_path: str):
    rows = []
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def analyze_units(rows, eeg_ids=("0x11", "0x12")):
    # Use payload_len from segmentation; focus on EEG ids
    by_id = {eid: [] for eid in eeg_ids}
    for row in rows:
        pkt_id = row.get("pkt_id")
        if pkt_id in by_id:
            try:
                L = int(row["payload_len"]) if row.get("payload_len") else 0
                by_id[pkt_id].append(L)
            except ValueError:
                continue

    summary = {}
    for eid, lens in by_id.items():
        if not lens:
            continue
        c = Counter(lens)
        mode_len, mode_cnt = max(c.items(), key=lambda kv: kv[1])

        def hist_mod(base):
            h = Counter(L % base for L in lens)
            mult_frac = sum(1 for L in lens if L % base == 0) / len(lens)
            k_counts = Counter(round(L / base) for L in lens)
            return h, mult_frac, k_counts

        h28, frac28, k28 = hist_mod(28)
        h56, frac56, k56 = hist_mod(56)

        summary[eid] = {
            "count": len(lens),
            "mode_len": mode_len,
            "mode_count": mode_cnt,
            "frac_mult_28": frac28,
            "frac_mult_56": frac56,
            "top_remainders_28": h28.most_common(5),
            "top_remainders_56": h56.most_common(5),
            "K28": k28.most_common(5),
            "K56": k56.most_common(5),
        }
    return summary


def print_summary(preset: str, summary):
    print(f"\nPreset {preset}")
    for eid, s in summary.items():
        print(
            f"  {eid}: count={s['count']}, mode_len={s['mode_len']} (x{s['mode_count']});"
            f" mult28={s['frac_mult_28']*100:.0f}%, mult56={s['frac_mult_56']*100:.0f}%"
        )
        print(f"    top mod28: {s['top_remainders_28']}")
        print(f"    top mod56: {s['top_remainders_56']}")
        print(f"    K28: {s['K28']}")
        print(f"    K56: {s['K56']}")


def main():
    p = argparse.ArgumentParser(description="Inspect EEG base units from segmented CSV")
    p.add_argument("--preset", required=True)
    p.add_argument("--csv", required=True)
    args = p.parse_args()

    rows = load_segmented(args.csv)
    summary = analyze_units(rows)
    print_summary(args.preset, summary)


if __name__ == "__main__":
    main()


