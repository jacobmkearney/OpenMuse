import argparse
import csv
import os
from collections import defaultdict, Counter
from statistics import mean, pstdev


def parse_packets_from_bin(data: bytes):
    """Yield (offset, packet_bytes) for each valid packet found.

    Each packet: first byte is declared length L, then L-1 following bytes.
    Stops when remaining length < L or L < 14.
    """
    offset = 0
    total = len(data)
    while offset < total:
        if offset + 1 > total:
            break
        declared_len = data[offset]
        # minimum header size
        if declared_len < 14:
            # move forward one byte to resync
            offset += 1
            continue
        if offset + declared_len > total:
            break
        pkt = data[offset : offset + declared_len]
        yield offset, pkt
        offset += declared_len


def analyze_bin_file(path: str):
    with open(path, "rb") as f:
        blob = f.read()

    results = []
    length_by_id = defaultdict(list)
    leftover_sizes_by_id = defaultdict(list)
    counts_by_id = Counter()

    for offset, pkt in parse_packets_from_bin(blob):
        pkt_len = pkt[0]
        pkt_id = pkt[9] if len(pkt) >= 10 else 0x00
        data = pkt[14:] if len(pkt) >= 14 else b""

        # Heuristic leftover sizing:
        # - If ACCGYRO id (0x47), primary payload = 36 bytes → leftover = data[36:]
        # - If Battery id (0x98), primary payload uses first 2 bytes (SoC) → leftover = data[2:]
        # - Otherwise, treat all data beyond header as leftover (unknown primary)
        if pkt_id == 0x47:
            primary = 36
            leftover = data[primary:] if len(data) >= primary else b""
        elif pkt_id == 0x98:
            primary = 2
            leftover = data[primary:] if len(data) >= primary else b""
        else:
            leftover = data

        counts_by_id[pkt_id] += 1
        length_by_id[pkt_id].append(pkt_len)
        leftover_sizes_by_id[pkt_id].append(len(leftover))

        results.append(
            {
                "offset": offset,
                "packet_id": pkt_id,
                "packet_length": pkt_len,
                "leftover_size": len(leftover),
            }
        )

    return results, counts_by_id, length_by_id, leftover_sizes_by_id


def write_csv(out_csv: str, preset: str, counts, lengths, leftover_sizes):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["packet_id", "count", "mean_len", "var_len", "leftover_mean", "leftover_hist"]) 
        for pkt_id in sorted(counts.keys()):
            lens = lengths[pkt_id]
            var = 0.0
            if len(lens) >= 2:
                # population variance
                m = mean(lens)
                var = mean([(x - m) ** 2 for x in lens])
            leftovers = leftover_sizes[pkt_id]
            lo_mean = mean(leftovers) if leftovers else 0.0
            # simple histogram bins for leftovers requested: 0, 28, 56, other
            bins = Counter()
            for s in leftovers:
                if s == 0:
                    bins["0"] += 1
                elif s == 28:
                    bins["28"] += 1
                elif s == 56:
                    bins["56"] += 1
                else:
                    bins[str(s)] += 1
            hist_str = ";".join(f"{k}:{v}" for k, v in sorted(bins.items()))
            w.writerow([f"0x{pkt_id:02x}", counts[pkt_id], f"{mean(lens):.2f}", f"{var:.2f}", f"{lo_mean:.2f}", hist_str])


def append_markdown(md_path: str, preset: str, counts, lengths, leftover_sizes):
    md_dir = os.path.dirname(md_path)
    if md_dir:
        os.makedirs(md_dir, exist_ok=True)
    with open(md_path, "a", encoding="utf-8") as f:
        f.write(f"\n\n### {preset}\n")
        f.write("\n")
        f.write("preset | packet_id | count | mean_len | var_len | leftover_mean | notes\n")
        f.write(":- | :-: | :-: | :-: | :-: | :-: | -\n")
        for pkt_id in sorted(counts.keys()):
            lens = lengths[pkt_id]
            var = 0.0
            if len(lens) >= 2:
                m = mean(lens)
                var = mean([(x - m) ** 2 for x in lens])
            lo_mean = mean(leftover_sizes[pkt_id]) if leftover_sizes[pkt_id] else 0.0
            notes = "stable" if var <= 4.0 else "variable"  # variance ≤ 2 bytes => var ≤ 4
            f.write(
                f"{preset} | 0x{pkt_id:02x} | {len(lens)} | {mean(lens):.2f} | {var:.2f} | {lo_mean:.2f} | {notes}\n"
            )


def main():
    p = argparse.ArgumentParser(description="Inspect packets in Muse .bin files")
    p.add_argument("--preset", required=True, help="Preset label (e.g., p1035)")
    p.add_argument("--infile", required=True, help="Input .bin file path")
    p.add_argument(
        "--csvout",
        required=False,
        help="CSV output path (default: data/{preset}/summary.csv)",
    )
    p.add_argument(
        "--mdout",
        required=False,
        help="Markdown output path to append (default: packet_analysis.md)",
    )
    args = p.parse_args()

    results, counts, lengths, leftover_sizes = analyze_bin_file(args.infile)

    # stdout summary
    print(f"Analyzed {sum(counts.values())} packets from {args.infile}")
    for pkt_id in sorted(counts.keys()):
        lens = lengths[pkt_id]
        m = mean(lens)
        var = 0.0
        if len(lens) >= 2:
            var = mean([(x - m) ** 2 for x in lens])
        print(f"  id=0x{pkt_id:02x}: count={counts[pkt_id]}, mean_len={m:.2f}, var_len={var:.2f}")

    csv_path = args.csvout or os.path.join("data", args.preset, "summary.csv")
    write_csv(csv_path, args.preset, counts, lengths, leftover_sizes)

    md_path = args.mdout or os.path.join("packet_analysis.md")
    append_markdown(md_path, args.preset, counts, lengths, leftover_sizes)


if __name__ == "__main__":
    main()


