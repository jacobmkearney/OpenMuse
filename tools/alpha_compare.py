import argparse
import numpy as np

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
        yield off, data[off : off + L]
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


def alpha_ratio(x: np.ndarray, fs: float) -> float:
    if len(x) < 8:
        return 0.0
    x = x.astype(np.float32)
    x = x - x.mean()
    freqs = np.fft.rfftfreq(len(x), d=1.0/fs)
    psd = np.abs(np.fft.rfft(x))**2
    def band(f1,f2):
        idx = (freqs>=f1) & (freqs<=f2)
        return float(psd[idx].sum())
    alpha = band(8,12)
    broad = band(1,40) + 1e-9
    return alpha / broad


def median_alpha_per_channel(blob: bytes, seconds: float, rate: float, stride_packets: int = 10, max_windows: int = 10):
    alphas = {}
    starts = 0
    for _ in range(max_windows):
        # collect ~seconds window starting at packet index starts
        needed = int(seconds * rate)
        locked = None
        series = None
        channels = None
        collected = 0
        pkt_idx = 0
        for _, pkt in parse_packets_from_bin(blob):
            if pkt_idx < starts:
                pkt_idx += 1
                continue
            if len(pkt) < 14:
                pkt_idx += 1
                continue
            pid = pkt[9]
            if pid not in EEG_IDS:
                pkt_idx += 1
                continue
            channels = EEG_IDS[pid]
            payload = pkt[14:]
            off = first_secondary_offset(payload)
            primary = payload if off is None else payload[:off]
            if len(primary) < 28:
                pkt_idx += 1
                continue
            if locked is None:
                locked = find_best_bit_offset(primary[:28], channels=channels)
            samples, _ = decode_eeg_payload(primary, channels=channels, block_bytes=28, forced_bit_offset=locked)
            if not samples:
                pkt_idx += 1
                continue
            if series is None:
                series = [[] for _ in range(channels)]
            for s in samples:
                for ch in range(channels):
                    series[ch].append(int(s[ch]))
                collected += 1
                if collected >= needed:
                    break
            if collected >= needed:
                break
            pkt_idx += 1
        if not series:
            break
        for ch in range(channels or 0):
            arr = np.asarray(series[ch], dtype=np.int32)
            alphas.setdefault(ch, []).append(alpha_ratio(arr, rate))
        starts += stride_packets
    return {ch: float(np.median(vals)) for ch, vals in alphas.items()}


def main():
    p = argparse.ArgumentParser(description='Compare alpha ratios between two recordings (closed - open)')
    p.add_argument('--preset', required=True)
    p.add_argument('--open', dest='file_open', required=True)
    p.add_argument('--closed', dest='file_closed', required=True)
    p.add_argument('--seconds', type=float, default=4.0)
    p.add_argument('--rate', type=float, default=256.0)
    p.add_argument('--mdout', default=None)
    args = p.parse_args()

    with open(args.file_open, 'rb') as f:
        blob_open = f.read()
    with open(args.file_closed, 'rb') as f:
        blob_closed = f.read()

    a_open = median_alpha_per_channel(blob_open, args.seconds, args.rate)
    a_closed = median_alpha_per_channel(blob_closed, args.seconds, args.rate)
    channels = sorted(set(a_open.keys()) | set(a_closed.keys()))
    deltas = {ch: a_closed.get(ch, 0.0) - a_open.get(ch, 0.0) for ch in channels}

    print(f'{args.preset} alpha medians (open):', a_open)
    print(f'{args.preset} alpha medians (closed):', a_closed)
    print(f'{args.preset} alpha delta (closed-open):', deltas)

    if args.mdout:
        with open(args.mdout, 'a', encoding='utf-8') as f:
            f.write(f"\n\n### Alpha delta (closed - open) — {args.preset}\n")
            f.write("ch | alpha_open | alpha_closed | delta\n")
            f.write(":-:|:-:|:-:|:-:\n")
            for ch in channels:
                f.write(f"{ch} | {a_open.get(ch,0.0):.3f} | {a_closed.get(ch,0.0):.3f} | {deltas[ch]:.3f}\n")


if __name__ == '__main__':
    main()


