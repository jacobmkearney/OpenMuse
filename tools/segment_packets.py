import argparse
import csv
import os
from collections import Counter


KNOWN_TAGS = {0x34, 0x35, 0x36, 0x47, 0x98, 0x11, 0x12}


def parse_packets_from_bin(data: bytes):
    offset = 0
    total = len(data)
    while offset < total:
        if offset + 1 > total:
            break
        declared_len = data[offset]
        if declared_len < 14:
            offset += 1
            continue
        if offset + declared_len > total:
            break
        pkt = data[offset : offset + declared_len]
        yield offset, pkt
        offset += declared_len


def scan_chunks(payload: bytes):
    """Scan payload for [TAG, LEN, DATA] sequences.

    Returns:
    - first_secondary_offset: index of first recognized [TAG,LEN,DATA]; None if not found
    - tags_seq: list of (tag, offset, length)
    """
    tags = []
    first_secondary_offset = None
    i = 0
    n = len(payload)
    while i + 2 <= n:
        tag = payload[i]
        length = payload[i + 1]
        if tag in KNOWN_TAGS and i + 2 + length <= n:
            if first_secondary_offset is None:
                first_secondary_offset = i
            tags.append((tag, i, length))
            i += 2 + length
        else:
            i += 1
    return first_secondary_offset, tags


def segment_file(preset: str, infile: str, out_csv: str):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(infile, "rb") as f:
        blob = f.read()

    rows = []
    packet_index = 0

    for _, pkt in parse_packets_from_bin(blob):
        total_len = pkt[0]
        header_len = 14 if len(pkt) >= 14 else len(pkt)
        pkt_id = pkt[9] if len(pkt) >= 10 else 0x00
        payload = pkt[header_len:total_len] if len(pkt) >= header_len else b""
        payload_len = len(payload)

        # Scan payload for secondary chunks
        first_secondary_offset, chunks = scan_chunks(payload)
        if first_secondary_offset is None:
            primary_eeg_len = payload_len
            leftover_len = 0
        else:
            primary_eeg_len = first_secondary_offset
            leftover_len = max(0, payload_len - primary_eeg_len)

        tags_seq = ",".join(f"{tag:02x}@+{off}" for tag, off, _ in chunks)

        rows.append(
            {
                "preset": preset,
                "packet_index": packet_index,
                "pkt_id": f"0x{pkt_id:02x}",
                "total_len": total_len,
                "header_len": header_len,
                "payload_len": payload_len,
                "first_secondary_offset": first_secondary_offset if first_secondary_offset is not None else -1,
                "primary_eeg_len": primary_eeg_len,
                "leftover_len": leftover_len,
                "tags_seq": tags_seq,
            }
        )

        packet_index += 1

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "preset",
                "packet_index",
                "pkt_id",
                "total_len",
                "header_len",
                "payload_len",
                "first_secondary_offset",
                "primary_eeg_len",
                "leftover_len",
                "tags_seq",
            ],
        )
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description="Per-packet segmentation (no decoding)")
    p.add_argument("--preset", required=True, help="Preset label (e.g., p1035)")
    p.add_argument("--infile", required=True, help="Input .bin file path")
    p.add_argument(
        "--csvout",
        required=False,
        help="CSV output path (default: data/{preset}/segmented.csv)",
    )
    args = p.parse_args()

    out_csv = args.csvout or os.path.join("data", args.preset, "segmented.csv")
    segment_file(args.preset, args.infile, out_csv)
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()


