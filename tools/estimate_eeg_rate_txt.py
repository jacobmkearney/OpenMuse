import argparse
import struct

EEG_IDS = {0x11: 4, 0x12: 8}
NON_EEG_TAGS = {0x34, 0x35, 0x36, 0x47, 0x98}


def parse_packets_from_payload(payload: bytes):
    off = 0
    n = len(payload)
    while off < n:
        if off + 14 > n:
            break
        L = payload[off]
        if L < 14 or off + L > n:
            break
        pkt = payload[off : off + L]
        yield off, pkt
        off += L


def first_secondary_offset(data: bytes):
    i = 0
    n = len(data)
    while i + 2 <= n:
        tag = data[i]
        ln = data[i + 1]
        if tag in NON_EEG_TAGS and i + 2 + ln <= n:
            return i
        i += 1
    return None


def estimate_rate_from_txt(infile: str):
    first_t = None
    last_t = None
    total_samples = 0
    eeg_id_used = None

    with open(infile, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue
            _, uuid, hexpayload = parts
            try:
                payload = bytes.fromhex(hexpayload)
            except Exception:
                continue
            for _, pkt in parse_packets_from_payload(payload):
                if len(pkt) < 14:
                    continue
                L = pkt[0]
                pid = pkt[9]
                if pid not in EEG_IDS:
                    continue
                t_ms = struct.unpack_from('<I', pkt, 2)[0]
                data = pkt[14:L]
                off = first_secondary_offset(data)
                primary = data if off is None else data[:off]
                blocks = len(primary) // 28
                samples = blocks * 2
                if samples <= 0:
                    continue
                eeg_id_used = pid
                if first_t is None:
                    first_t = t_ms
                last_t = t_ms
                total_samples += samples

    if first_t is None or last_t is None or total_samples == 0:
        return None
    duration_s = max(1e-6, (last_t - first_t) / 1000.0)
    sr_hz = total_samples / duration_s
    return {
        'eeg_id': f'0x{eeg_id_used:02x}' if eeg_id_used is not None else None,
        'total_samples': total_samples,
        'duration_s': duration_s,
        'rate_hz': sr_hz,
    }


def main():
    p = argparse.ArgumentParser(description='Estimate EEG sample rate from text recording')
    p.add_argument('--infile', required=True)
    args = p.parse_args()

    res = estimate_rate_from_txt(args.infile)
    if not res:
        print('No EEG detected')
        return
    print(f"EEG id {res['eeg_id']}: samples={res['total_samples']} over {res['duration_s']:.2f}s → rate≈{res['rate_hz']:.1f} Hz")


if __name__ == '__main__':
    main()


