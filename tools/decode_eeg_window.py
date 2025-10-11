import argparse
import statistics as st
from typing import List

from muse_eeg_unpack import decode_eeg_payload, find_best_bit_offset


EEG_IDS = {0x11: 4, 0x12: 8}
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


def decode_window(infile: str, seconds: float, freq_hz: float, preset: str):
    """Decode roughly 'seconds' of EEG given locked offset from early packets.

    freq_hz: nominal EEG sampling rate per channel (256 for Muse). We will stop
    after ~seconds * freq_hz samples.
    """
    with open(infile, "rb") as f:
        blob = f.read()

    needed = int(seconds * freq_hz)
    per_ch_series: List[List[int]] = []
    channels = None
    locked_offset = None
    collected = 0

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

        if locked_offset is None:
            locked_offset = find_best_bit_offset(primary[:28], channels=channels)

        samples, _ = decode_eeg_payload(primary, channels=channels, block_bytes=28, forced_bit_offset=locked_offset)
        if not samples:
            continue

        # Initialize per-channel lists
        if not per_ch_series:
            per_ch_series = [[] for _ in range(channels)]
        for s in samples:
            for ch in range(channels):
                per_ch_series[ch].append(s[ch])
                collected += 1
                if collected >= needed:
                    break
            if collected >= needed:
                break
        if collected >= needed:
            break

    # Compute continuity metrics
    metrics = []
    for ch in range(channels or 0):
        series = per_ch_series[ch]
        diffs = [abs(series[i + 1] - series[i]) for i in range(len(series) - 1)]
        p90 = sorted(diffs)[int(0.9 * len(diffs))] if diffs else 0
        rng = (min(series), max(series)) if series else (0, 0)
        metrics.append((p90, rng, len(series)))
    print(f"{preset}: locked bit_off={locked_offset} ch={channels} seconds≈{seconds}")
    for ch, (p90, rng, ln) in enumerate(metrics):
        print(f"  ch{ch}: n={ln}, p90|Δ|≈{p90:.1f}, range={rng}")


def main():
    p = argparse.ArgumentParser(description="Decode ~window of EEG with locked bit offset")
    p.add_argument("--preset", required=True)
    p.add_argument("--infile", required=True)
    p.add_argument("--seconds", type=float, default=3.0)
    p.add_argument("--rate", type=float, default=256.0)
    args = p.parse_args()

    decode_window(args.infile, args.seconds, args.rate, args.preset)


if __name__ == "__main__":
    main()


