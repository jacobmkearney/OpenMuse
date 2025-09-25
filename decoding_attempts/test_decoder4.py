import os
import re
import sys
from collections import Counter, defaultdict
from math import sqrt

# ------------------------
# Constants / Regexps
# ------------------------
TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}\+\d{2}:\d{2}$")
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
HEX_RE = re.compile(r"^[0-9a-f]+$")

# Parameters
MAX_LINES_SAMPLE = 5000  # how many lines to sample for offset discovery
MAX_OFFSET_SCAN = 64  # how many leading bytes (offsets) to examine
UNPACK_TEST_BYTES = 512  # bytes used when attempting to unpack bit-packed samples


# ------------------------
# Utility functions
# ------------------------
def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield line.rstrip("\n")


def bytes_from_hex(h):
    try:
        return bytes.fromhex(h)
    except Exception:
        return None


def validate_lines(path, max_lines=None):
    errors = []
    counts = Counter()
    n = 0
    for line in read_lines(path):
        n += 1
        if max_lines and n > max_lines:
            break
        parts = line.split("\t")
        if len(parts) != 3:
            counts["bad_field_count"] += 1
            if len(errors) < 20:
                errors.append((n, "field_count", line))
            continue
        ts, uid, payload = parts
        if not TS_RE.match(ts):
            counts["bad_timestamp"] += 1
            if len(errors) < 20:
                errors.append((n, "bad_timestamp", ts))
        if not UUID_RE.match(uid):
            counts["bad_uuid"] += 1
            if len(errors) < 20:
                errors.append((n, "bad_uuid", uid))
        if len(payload) % 2 != 0 or not HEX_RE.match(payload):
            counts["bad_hex"] += 1
            if len(errors) < 20:
                errors.append((n, "bad_hex", payload[:80]))
        counts["total_lines"] += 1
    return {"counts": counts, "errors": errors}


# Generic subpacket parser with configurable length-field width & endianness and header offset
def try_subpacket_by_length_at_offset(b, offset=0, length_bytes=1, little_endian=True):
    """
    Parse b starting at offset. Each subpacket begins with length field (length_bytes bytes),
    interpreted as unsigned integer (little_endian if True). Returns list of subpacket-data bytes
    or None if parsing fails anywhere.
    """
    i = offset
    Lb = len(b)
    parts = []
    while i < Lb:
        # ensure we can read length field
        if i + length_bytes > Lb:
            return None
        # read length field
        if length_bytes == 1:
            L = b[i]
        elif length_bytes == 2:
            if little_endian:
                L = b[i] | (b[i + 1] << 8)
            else:
                L = (b[i] << 8) | b[i + 1]
        elif length_bytes == 3:
            # 3-byte little-endian (common in some BLE fragmentation schemes)
            if not little_endian:
                # not implementing BE 3-byte; if needed extend here
                return None
            L = b[i] | (b[i + 1] << 8) | (b[i + 2] << 16)
        else:
            return None
        hdr_len = length_bytes
        if i + hdr_len + L > Lb:
            return None
        parts.append(b[i + hdr_len : i + hdr_len + L])
        i += hdr_len + L
    return parts


def unpack_bits_le(b, bits):
    if bits <= 0 or bits > 32:
        raise ValueError("bits must be 1..32")
    bitbuf = 0
    bitlen = 0
    out = []
    val_mask = (1 << bits) - 1
    sign_mask = 1 << (bits - 1)
    for byte in b:
        bitbuf |= byte << bitlen
        bitlen += 8
        while bitlen >= bits:
            v = bitbuf & val_mask
            bitbuf >>= bits
            bitlen -= bits
            if v & sign_mask:
                v = v - (1 << bits)
            out.append(v)
    return out


def stats_of_list(xs):
    if not xs:
        return {"n": 0}
    n = len(xs)
    s = sum(xs)
    mean = s / n
    var = sum((x - mean) ** 2 for x in xs) / n
    std = var**0.5
    return {"n": n, "min": min(xs), "max": max(xs), "mean": mean, "std": std}


def try_unpack_region(region_bytes):
    out = {}
    for bits in (14, 16, 20):
        try:
            samples = unpack_bits_le(region_bytes, bits)
            s = stats_of_list(samples[:20000])
            out[bits] = s
        except Exception as e:
            out[bits] = {"error": str(e)}
    return out


def process_file(path):
    print(f"\n--- Processing {path} ---")
    v = validate_lines(path, max_lines=20000)
    counts = v["counts"]
    print("Lines:", counts.get("total_lines", 0))
    if any(
        counts[k] for k in ("bad_field_count", "bad_timestamp", "bad_uuid", "bad_hex")
    ):
        print(
            "Validation issues:",
            {
                k: counts[k]
                for k in ("bad_field_count", "bad_timestamp", "bad_uuid", "bad_hex")
                if counts.get(k)
            },
        )
    else:
        print("Line format: OK")

    # sample payloads
    payloads = []
    for i, line in enumerate(read_lines(path)):
        if i >= MAX_LINES_SAMPLE:
            break
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        payloads.append(parts[2])
    if not payloads:
        print("No payloads found.")
        return

    # Try parsing at offsets 0..MAX_OFFSET_SCAN-1 for length widths 1,2,3
    results = {}
    for offset in range(0, MAX_OFFSET_SCAN):
        for length_bytes in (1, 2, 3):
            if length_bytes == 2:
                for le in (True, False):
                    key = (offset, length_bytes, "LE" if le else "BE")
                    ok = 0
                    total = 0
                    for h in payloads:
                        b = bytes_from_hex(h)
                        if not b:
                            continue
                        total += 1
                        parts = try_subpacket_by_length_at_offset(
                            b,
                            offset=offset,
                            length_bytes=length_bytes,
                            little_endian=le,
                        )
                        if parts is not None:
                            ok += 1
                    results[key] = (ok, total)
            elif length_bytes in (1, 3):
                key = (offset, length_bytes, "LE")
                ok = 0
                total = 0
                for h in payloads:
                    b = bytes_from_hex(h)
                    if not b:
                        continue
                    total += 1
                    parts = try_subpacket_by_length_at_offset(
                        b, offset=offset, length_bytes=length_bytes, little_endian=True
                    )
                    if parts is not None:
                        ok += 1
                results[key] = (ok, total)

    # Summarize top candidates
    ranked = sorted(
        results.items(), key=lambda kv: (-kv[1][0], kv[1][1])
    )  # more ok first
    print(
        "\nTop parsing candidates (offset, length_bytes, endian) -> success/total (percent):"
    )
    for k, (ok, tot) in ranked[:12]:
        pct = 100.0 * ok / tot if tot else 0.0
        print(f"  {k} -> {ok}/{tot} ({pct:.1f}%)")

    # If no candidate gives >0, print a few diagnostics and try an alternative heuristics:
    best_ok = ranked[0][1][0] if ranked else 0
    if best_ok == 0:
        print(
            "\nNo length-prefix parsing candidate succeeded on any payload in the sample window."
        )
        print("Diagnostics:")
        print(
            "  - show first 8 bytes hex of first 8 payloads (to inspect header patterns)"
        )
        for i, h in enumerate(payloads[:8]):
            b = bytes_from_hex(h)
            print(f"   [{i}] {b[:8].hex()}  len={len(b)}")
        print("\n  - show byte-frequency for first 24 offsets across the sample window")
        freq = [Counter() for _ in range(24)]
        for h in payloads:
            b = bytes_from_hex(h)
            if not b:
                continue
            for off in range(min(24, len(b))):
                freq[off][b[off]] += 1
        print(" offset: top-3")
        for off in range(min(24, len(freq))):
            top = freq[off].most_common(3)
            top_str = ", ".join(f"{v}:{c}" for v, c in top)
            print(f"  {off:2d}: {top_str}")
        # Try an alternative: maybe subpackets are separated by a constant marker byte sequence.
        # Search for a repeated 2-byte separator in many payloads.
        sep2 = Counter()
        for h in payloads:
            b = bytes_from_hex(h)
            if not b or len(b) < 6:
                continue
            # collect all adjacent 2-byte pairs
            for i in range(len(b) - 1):
                sep2[(b[i], b[i + 1])] += 1
        print("\nTop 10 2-byte pairs in payloads (candidate separators):")
        for (a, b), ct in sep2.most_common(10):
            print(f"  {a:02x}{b:02x} : {ct}")
        print(
            "\nIf you want, rerun with a larger sample window or share a small example payload (first 2–4 lines)."
        )
    else:
        # If a candidate exists, pick best and show more detail, including unpacking region
        best_key = ranked[0][0]
        ok, tot = results[best_key]
        pct = 100.0 * ok / tot if tot else 0.0
        print(f"\nBest candidate: {best_key} -> {ok}/{tot} ({pct:.1f}%)")
        offset, length_bytes, endian = best_key
        le = True if endian == "LE" else False
        # show distribution of subpacket lengths for this best candidate
        sublen_counter = Counter()
        example_first = Counter()
        for h in payloads:
            b = bytes_from_hex(h)
            if not b:
                continue
            parts = try_subpacket_by_length_at_offset(
                b, offset=offset, length_bytes=length_bytes, little_endian=le
            )
            if parts is not None:
                for p in parts:
                    sublen_counter[len(p)] += 1
                example_first[bytes(b[:8]).hex()] += 1
        print("\nSubpacket length histogram (best candidate) top 12:")
        for ln, ct in sublen_counter.most_common(12):
            print(f"  len={ln:4d}  count={ct:6d}")
        print("\nMost common first-8-bytes (sample window) top 8:")
        for k, c in example_first.most_common(8):
            print(f"  {k}  count={c}")

        # pick a representative subpacket region to try unpacking samples
        rep_region = None
        for h in payloads:
            b = bytes_from_hex(h)
            if not b:
                continue
            parts = try_subpacket_by_length_at_offset(
                b, offset=offset, length_bytes=length_bytes, little_endian=le
            )
            if parts:
                rep_region = parts[0]
                break
        if rep_region is None:
            print(
                "Could not obtain a representative subpacket despite candidate being best; aborting unpack trial."
            )
            return
        # try unpacking
        region = rep_region[:UNPACK_TEST_BYTES]
        unpack_stats = try_unpack_region(region)
        print(f"\nUnpack trial stats for first {len(region)} bytes of first subpacket:")
        for bits, s in unpack_stats.items():
            if "error" in s:
                print(f"  {bits:2d}-bit: error {s['error']}")
            else:
                print(
                    f"  {bits:2d}-bit: n={s['n']}, min={s['min']}, max={s['max']}, mean={s['mean']:.2f}, std={s['std']:.2f}"
                )


# ------------------------
# Script entrypoint
# ------------------------
if __name__ == "__main__":
    data_dir = "./data_raw"
    files = sorted(os.listdir(data_dir))
    for f in files:
        path = os.path.join(data_dir, f)
        process_file(path)
