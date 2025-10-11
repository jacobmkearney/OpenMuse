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
    payload_len_by_id = defaultdict(list)
    leftover_sizes_by_id = defaultdict(list)
    counts_by_id = Counter()
    # EEG remainder tracking
    remainder28_by_id = defaultdict(list)
    remainder56_by_id = defaultdict(list)
    # In-payload chunk scanning (best-effort TAG, LEN, DATA)
    chunk_tag_counts_by_id = defaultdict(Counter)
    chunk_tag_len_hist_by_id = defaultdict(lambda: defaultdict(Counter))

    for offset, pkt in parse_packets_from_bin(blob):
        pkt_len = pkt[0]
        pkt_id = pkt[9] if len(pkt) >= 10 else 0x00
        data = pkt[14:] if len(pkt) >= 14 else b""
        payload_len = max(0, len(pkt) - 14)

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
        payload_len_by_id[pkt_id].append(payload_len)
        leftover_sizes_by_id[pkt_id].append(len(leftover))
        # EEG remainder stats only for EEG IDs (0x11, 0x12)
        if pkt_id in (0x11, 0x12):
            remainder28_by_id[pkt_id].append(payload_len % 28 if 28 else 0)
            remainder56_by_id[pkt_id].append(payload_len % 56 if 56 else 0)

        # Chunk scanning: treat payload as potential sequence of [TAG, LEN, DATA]
        # to detect co-packed data signatures (e.g., optics/IMU/battery)
        i = 0
        while i + 2 <= len(data):
            tag = data[i]
            length_byte = data[i + 1]
            # Known tag bytes of interest
            if tag in (0x34, 0x35, 0x36, 0x47, 0x98, 0x11, 0x12):
                end = i + 2 + length_byte
                if end <= len(data):
                    chunk_tag_counts_by_id[pkt_id][tag] += 1
                    chunk_tag_len_hist_by_id[pkt_id][tag][length_byte] += 1
                    i = end
                    continue
            i += 1

        results.append(
            {
                "offset": offset,
                "packet_id": pkt_id,
                "packet_length": pkt_len,
                "payload_length": payload_len,
                "leftover_size": len(leftover),
            }
        )

    return (
        results,
        counts_by_id,
        length_by_id,
        payload_len_by_id,
        leftover_sizes_by_id,
        remainder28_by_id,
        remainder56_by_id,
        chunk_tag_counts_by_id,
        chunk_tag_len_hist_by_id,
    )


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


def _compute_eeg_notes(payload_len_by_id, remainder28_by_id, remainder56_by_id):
    notes = {}
    # EEG4 id 0x11, EEG8 id 0x12 (from decode.py TAGS mapping)
    candidates = {0x11: [28, 56], 0x12: [28, 56]}
    for eeg_id, bases in candidates.items():
        if eeg_id not in payload_len_by_id:
            continue
        lens = payload_len_by_id[eeg_id]
        best = None
        for base in bases:
            if base <= 0:
                continue
            zeros = sum(1 for L in lens if L % base == 0)
            frac = zeros / len(lens) if lens else 0.0
            # Use mode payload to estimate units
            c = Counter(lens)
            mode_len, _ = max(c.items(), key=lambda kv: kv[1])
            units = mode_len / base if base else 0
            # Estimate samples assuming base unit encodes 2 samples
            est_samples = int(round(units * 2))
            # Remainder concentration summary
            rems = remainder28_by_id if base == 28 else remainder56_by_id
            rem_counts = Counter(rems.get(eeg_id, []))
            top_rems = ", ".join(
                f"{r}:{cnt}" for r, cnt in rem_counts.most_common(3)
            ) if rem_counts else ""
            summary = (
                f"base {base}B: {frac*100:.0f}% multiples; mode {mode_len}B → ~{units:.1f} units (~{est_samples} samples); "
                f"top remainders [{top_rems}]"
            )
            if best is None or frac > best[0]:
                best = (frac, summary)
        if best is not None:
            notes[eeg_id] = best[1]
    return notes


def append_markdown(md_path: str, preset: str, counts, lengths, leftover_sizes, payload_lens, remainder28, remainder56, chunk_tag_counts, chunk_tag_len_hist):
    md_dir = os.path.dirname(md_path)
    if md_dir:
        os.makedirs(md_dir, exist_ok=True)
    eeg_notes = _compute_eeg_notes(payload_lens, remainder28, remainder56)
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
            notes = eeg_notes.get(pkt_id)
            if not notes:
                notes = "stable" if var <= 4.0 else "variable"
            f.write(
                f"{preset} | 0x{pkt_id:02x} | {len(lens)} | {mean(lens):.2f} | {var:.2f} | {lo_mean:.2f} | {notes}\n"
            )

        # Append remainder histograms for EEG IDs
        for eeg_id in (0x11, 0x12):
            if eeg_id in payload_lens:
                r28 = Counter(remainder28.get(eeg_id, []))
                r56 = Counter(remainder56.get(eeg_id, []))
                def fmt_hist(c):
                    items = sorted(c.items())
                    return ", ".join(f"{k}:{v}" for k, v in items[:10]) if items else ""
                f.write("\n")
                f.write(
                    f"Remainders for 0x{eeg_id:02x}: mod28 [{fmt_hist(r28)}], mod56 [{fmt_hist(r56)}]\n"
                )

        # Append detected in-payload chunk summaries (top tags)
        f.write("\nDetected co-packed chunks (TAG:count) by primary packet_id:\n\n")
        for pid in sorted(chunk_tag_counts.keys()):
            tag_counts = chunk_tag_counts[pid]
            if not tag_counts:
                continue
            top = ", ".join(
                f"0x{tag:02x}:{cnt}" for tag, cnt in tag_counts.most_common(5)
            )
            f.write(f"  - 0x{pid:02x}: {top}\n")

        # Final EEG summary table (per preset)
        def mode(items):
            if not items:
                return None
            c = Counter(items)
            return max(c.items(), key=lambda kv: kv[1])[0]

        def mult_str(n: int, base: int) -> str:
            if base <= 0:
                return "—"
            return f"{base}×{n//base}" if (n % base) == 0 else "—"

        f.write("\n\nEEG summary (dominant payload and samples-per-packet estimate)\n\n")
        f.write("preset | EEG id | dominant payload length | multiple-of-28/56 | est samples/packet\n")
        f.write(":- | :-: | :-: | :-: | :-:\n")
        for eeg_id, chans in ((0x11, 4), (0x12, 8)):
            if eeg_id not in payload_lens:
                continue
            dom = mode(payload_lens[eeg_id])
            if dom is None:
                continue
            m28 = mult_str(dom, 28)
            m56 = mult_str(dom, 56)
            mults = m56 if m56 != "—" else (m28 if m28 != "—" else "—")
            # Estimate samples: bits = dom*8; per-sample bits = chans*14
            est = round((dom * 8) / (chans * 14))
            f.write(f"{preset} | 0x{eeg_id:02x} | {dom} | {mults} | {est}\n")


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

    (
        results,
        counts,
        lengths,
        payload_lens,
        leftover_sizes,
        remainder28,
        remainder56,
        chunk_tag_counts,
        chunk_tag_len_hist,
    ) = analyze_bin_file(args.infile)

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
    append_markdown(
        md_path,
        args.preset,
        counts,
        lengths,
        leftover_sizes,
        payload_lens,
        remainder28,
        remainder56,
        chunk_tag_counts,
        chunk_tag_len_hist,
    )


if __name__ == "__main__":
    main()


