import argparse
import numpy as np
from typing import List, Tuple

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


def collect_series(infile: str, seconds: float, rate: float) -> Tuple[List[List[int]], int, int]:
    with open(infile, 'rb') as f:
        blob = f.read()

    needed = int(seconds * rate)
    channels = None
    locked = None
    series: List[List[int]] = []
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
        if locked is None:
            locked = find_best_bit_offset(primary[:28], channels=channels)
        samples, _ = decode_eeg_payload(primary, channels=channels, block_bytes=28, forced_bit_offset=locked)
        if not samples:
            continue
        if not series:
            series = [[] for _ in range(channels)]
        for s in samples:
            for ch in range(channels):
                series[ch].append(int(s[ch]))
            collected += 1
            if collected >= needed:
                break
        if collected >= needed:
            break

    return series, channels or 0, locked or 0


def alpha_ratio(x: List[int], fs: float) -> float:
    if len(x) < 8:
        return 0.0
    xf = np.asarray(x, dtype=np.float32)
    xf = xf - np.mean(xf)
    N = len(xf)
    freqs = np.fft.rfftfreq(N, d=1.0/fs)
    psd = np.abs(np.fft.rfft(xf))**2
    def band(f1,f2):
        idx = (freqs>=f1) & (freqs<=f2)
        return float(np.sum(psd[idx]))
    alpha = band(8,12)
    broad = band(1,40) + 1e-9
    return alpha/broad


def blink_strength(x: List[int]) -> float:
    if len(x) < 8:
        return 0.0
    x = np.asarray(x, dtype=np.float32)
    z = (x - np.mean(x)) / (np.std(x) + 1e-6)
    # Blink heuristic: top 0.5% absolute z amplitude
    thr = np.percentile(np.abs(z), 99.5)
    idx = np.where(np.abs(z) >= thr)[0]
    return float(np.median(np.abs(x[idx]))) if idx.size else 0.0


def estimate_pairs(series: List[List[int]], fs: float) -> Tuple[List[Tuple[int,int,float]], List[Tuple[int,float]], List[Tuple[int,float]]]:
    ch = len(series)
    # Correlation matrix
    corr = np.zeros((ch, ch), dtype=np.float32)
    arr = [np.asarray(s, dtype=np.float32) for s in series]
    for i in range(ch):
        for j in range(i, ch):
            if len(arr[i]) < 2 or len(arr[j]) < 2:
                r = 0.0
            else:
                a = arr[i] - np.mean(arr[i])
                b = arr[j] - np.mean(arr[j])
                den = (np.std(a) * np.std(b) + 1e-9)
                r = float(np.sum(a*b) / (len(a) * den))
            corr[i,j] = corr[j,i] = r
    # Greedy pairing by highest correlation off-diagonal
    used = set()
    pairs = []
    while len(used) < ch-1:
        best = None
        for i in range(ch):
            if i in used: continue
            for j in range(i+1, ch):
                if j in used: continue
                v = corr[i,j]
                if best is None or v > best[2]:
                    best = (i,j,v)
        if best is None: break
        i,j,v = best
        pairs.append(best)
        used.add(i); used.add(j)

    # Blink strength and alpha ratio per channel
    blink = [(i, blink_strength(series[i])) for i in range(ch)]
    alpha = [(i, alpha_ratio(series[i], fs)) for i in range(ch)]
    blink.sort(key=lambda x: -x[1])
    alpha.sort(key=lambda x: -x[1])
    return pairs, blink, alpha


def main():
    p = argparse.ArgumentParser(description='Estimate channel pairing and markers (blink/alpha)')
    p.add_argument('--preset', required=True)
    p.add_argument('--infile', required=True)
    p.add_argument('--seconds', type=float, default=10.0)
    p.add_argument('--rate', type=float, default=256.0)
    args = p.parse_args()

    series, ch, bitoff = collect_series(args.infile, args.seconds, args.rate)
    if ch == 0:
        print('No EEG decoded.')
        return
    pairs, blink, alpha = estimate_pairs(series, args.rate)

    print(f'{args.preset}: ch={ch} bit_off={bitoff}')
    print('Pairing by correlation (i,j,r):')
    for i,j,r in pairs:
        print(f'  ({i},{j}) r={r:.2f}')
    print('Blink strength (desc):')
    print('  ' + ', '.join(f'ch{k}:{v:.1f}' for k,v in blink))
    print('Alpha ratio (desc):')
    print('  ' + ', '.join(f'ch{k}:{v:.3f}' for k,v in alpha))
    # Heuristic labeling hints
    frontal = [k for k,_ in blink[:2]]
    posterior = [k for k,_ in alpha[:2]]
    print(f'Heuristic frontal candidates: {frontal}; posterior candidates: {posterior}')


if __name__ == '__main__':
    main()


