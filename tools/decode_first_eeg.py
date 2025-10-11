import argparse
import os
import statistics as st

from muse_eeg_unpack import decode_eeg_payload, find_best_bit_offset


EEG_IDS = {0x11: 4, 0x12: 8}
# Consider ONLY non-EEG tags as secondary boundaries for primary EEG slicing
NON_EEG_TAGS = {0x34, 0x35, 0x36, 0x47, 0x98}


def parse_packets_from_bin(data: bytes):
    off = 0
    n = len(data)
    while off < n:
        if off + 1 > n:
            break
        L = data[off]
        if L < 14:
            off += 1
            continue
        if off + L > n:
            break
        pkt = data[off : off + L]
        yield off, pkt
        off += L


def first_secondary_offset(payload: bytes):
    i = 0
    n = len(payload)
    while i + 2 <= n:
        tag = payload[i]
        ln = payload[i + 1]
        if tag in NON_EEG_TAGS and i + 2 + ln <= n:
            return i
        i += 1
    return None


def decode_first_eeg_packets(preset: str, infile: str, max_packets: int = 10, lock_first_n: int = 10):
    with open(infile, "rb") as f:
        blob = f.read()

    decoded = 0
    locked_offset = None
    tested = 0
    for _, pkt in parse_packets_from_bin(blob):
        if len(pkt) < 14:
            continue
        pkt_id = pkt[9]
        if pkt_id not in EEG_IDS:
            continue
        channels = EEG_IDS[pkt_id]
        payload = pkt[14:]
        off = first_secondary_offset(payload)
        primary = payload if off is None else payload[:off]
        if len(primary) < 28:
            continue

        # Establish locked bit offset using first full block of first lock_first_n packets
        if locked_offset is None and tested < lock_first_n and len(primary) >= 28:
            locked_offset = find_best_bit_offset(primary[:28], channels=channels)
        tested += 1

        samples, bit_offset = decode_eeg_payload(
            primary, channels=channels, block_bytes=28, forced_bit_offset=locked_offset
        )
        if not samples:
            continue

        # Compute simple continuity metrics per channel
        num_samples = len(samples)
        # Interleave into per-channel series
        per_ch = list(zip(*samples))  # channels -> list of len num_samples
        ch_stats = []
        for ch in range(channels):
            series = list(per_ch[ch])
            diffs = [abs(series[i + 1] - series[i]) for i in range(len(series) - 1)]
            diffs_sorted = sorted(diffs) if diffs else []
            p90 = diffs_sorted[int(0.9 * len(diffs_sorted))] if diffs_sorted else 0
            rng = (min(series), max(series)) if series else (0, 0)
            ch_stats.append((p90, rng))

        print(f"{preset} EEG id=0x{pkt_id:02x} ch={channels} bit_off={bit_offset} samples={num_samples}")
        # Summarize across channels
        avg_p90 = st.mean(s[0] for s in ch_stats) if ch_stats else 0
        min_rng = min(s[1][0] for s in ch_stats) if ch_stats else 0
        max_rng = max(s[1][1] for s in ch_stats) if ch_stats else 0
        print(f"  continuity p90(abs diff)≈{avg_p90:.1f}, value range≈[{min_rng},{max_rng}]")

        decoded += 1
        if decoded >= max_packets:
            break


def main():
    p = argparse.ArgumentParser(description="Decode first EEG packets and print sanity metrics")
    p.add_argument("--preset", required=True)
    p.add_argument("--infile", required=True)
    p.add_argument("--max", type=int, default=10)
    args = p.parse_args()

    decode_first_eeg_packets(args.preset, args.infile, args.max)


if __name__ == "__main__":
    main()


