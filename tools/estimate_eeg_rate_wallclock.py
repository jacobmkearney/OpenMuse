import argparse
from datetime import datetime
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


def estimate_rate_wallclock(infile: str):
    first_ts = None
    last_ts = None
    total_samples = 0
    eeg_buffer = bytearray()
    eeg_id_used = None

    with open(infile, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue
            ts, uuid, hexpayload = parts
            try:
                t = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                payload = bytes.fromhex(hexpayload)
            except Exception:
                continue
            line_samples = 0
            for _, pkt in parse_packets_from_payload(payload):
                if len(pkt) < 14:
                    continue
                L = pkt[0]
                pid = pkt[9]
                if pid not in EEG_IDS:
                    continue
                data = pkt[14:L]
                # Extract EEG bytes across entire payload by skipping non-EEG chunks
                i = 0
                n = len(data)
                while i < n:
                    if i + 2 <= n and data[i] in NON_EEG_TAGS and i + 2 + (data[i+1]) <= n:
                        i += 2 + data[i+1]
                        continue
                    if i + 2 <= n and data[i] in EEG_IDS and i + 2 + (data[i+1]) <= n:
                        ln = data[i+1]
                        eeg_buffer.extend(data[i+2:i+2+ln])
                        i += 2 + ln
                        continue
                    # Treat untagged byte as EEG
                    eeg_buffer.append(data[i])
                    i += 1
                # Count full 28B blocks from rolling buffer
                blocks = len(eeg_buffer) // 28
                if blocks > 0:
                    line_samples += blocks * 2
                    eeg_buffer = eeg_buffer[blocks*28:]
                eeg_id_used = pid
            if line_samples > 0:
                if first_ts is None:
                    first_ts = t
                last_ts = t
                total_samples += line_samples

    if first_ts is None or last_ts is None or total_samples == 0:
        return None
    duration_s = max(1e-6, (last_ts - first_ts).total_seconds())
    sr_hz = total_samples / duration_s
    return {
        'eeg_id': f'0x{eeg_id_used:02x}' if eeg_id_used is not None else None,
        'total_samples': total_samples,
        'duration_s': duration_s,
        'rate_hz': sr_hz,
    }


def main():
    p = argparse.ArgumentParser(description='Estimate EEG sample rate using wall-clock timestamps in text recording')
    p.add_argument('--infile', required=True)
    args = p.parse_args()

    res = estimate_rate_wallclock(args.infile)
    if not res:
        print('No EEG detected')
        return
    print(f"Wall-clock EEG id {res['eeg_id']}: samples={res['total_samples']} over {res['duration_s']:.2f}s → rate≈{res['rate_hz']:.1f} Hz")


if __name__ == '__main__':
    main()


