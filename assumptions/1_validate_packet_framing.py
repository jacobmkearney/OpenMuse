import argparse
import os
from collections import Counter, defaultdict
from statistics import mean


def parse_packets_from_bin(data: bytes):
    """Yield (offset, packet_bytes) for each candidate packet found.

    Packet framing rule under test:
      - First byte is declared length L
      - Packet spans L bytes total
      - Minimum plausible header length is 14 bytes
    We walk by declared lengths; if L < 14 we advance by 1 byte to resync.
    """
    offset = 0
    total = len(data)
    while offset < total:
        if offset + 1 > total:
            break
        declared_len = data[offset]
        if declared_len < 14:
            # Not a plausible packet start; advance by one to attempt resync
            offset += 1
            continue
        if offset + declared_len > total:
            # Truncated tail; stop
            break
        pkt = data[offset : offset + declared_len]
        yield offset, pkt
        offset += declared_len


def analyze_packet_framing(blob: bytes):
    """Compute Stage A diagnostics and invariants over a binary blob.

    Returns a dict with histograms and anomaly counts.
    """
    id_counts = Counter()
    len_lists = defaultdict(list)
    align_zero_count = 0
    align_nonzero = Counter()  # byte13 value -> count
    bad_id_counts = Counter()  # ids considered impossible (0x00, 0xff)
    total_packets = 0

    for _, pkt in parse_packets_from_bin(blob):
        total_packets += 1
        pkt_len = pkt[0]
        pkt_id = pkt[9] if len(pkt) >= 10 else 0x00
        byte_13 = pkt[13] if len(pkt) >= 14 else 0xFF

        id_counts[pkt_id] += 1
        len_lists[pkt_id].append(pkt_len)

        if byte_13 == 0x00:
            align_zero_count += 1
        else:
            align_nonzero[byte_13] += 1

        if pkt_id in (0x00, 0xFF):
            bad_id_counts[pkt_id] += 1

    results = {
        "total_packets": total_packets,
        "id_counts": id_counts,
        "len_lists": len_lists,
        "align_zero_count": align_zero_count,
        "align_nonzero": align_nonzero,
        "bad_id_counts": bad_id_counts,
    }
    return results


def assert_invariants(diagnostics: dict, strict: bool = True) -> None:
    """Fail loudly if Stage A invariants are violated.

    Invariants tested (packet framing stage):
      1) At least one packet found
      2) No impossible IDs (0x00, 0xFF)
      3) Alignment marker byte[13] is 0x00 for (nearly) all packets
         - strict: require 100%
         - non-strict: require >= 99%
    """
    total = diagnostics["total_packets"]
    if total == 0:
        raise AssertionError("No packets parsed; packet framing failed.")

    bad_ids_total = sum(diagnostics["bad_id_counts"].values())
    if bad_ids_total > 0:
        details = ", ".join(
            f"0x{k:02x}={v}" for k, v in diagnostics["bad_id_counts"].items()
        )
        raise AssertionError(f"Found impossible packet IDs: {details}")

    align_zero = diagnostics["align_zero_count"]
    align_nonzero_total = sum(diagnostics["align_nonzero"].values())
    align_rate = align_zero / total if total else 0.0
    if strict:
        if align_nonzero_total != 0:
            details = ", ".join(
                f"0x{k:02x}={v}" for k, v in diagnostics["align_nonzero"].items()
            )
            raise AssertionError(
                f"Alignment marker byte[13] not 0x00 for {align_nonzero_total}/{total} packets: {details}"
            )
    else:
        if align_rate < 0.99:
            raise AssertionError(
                f"Alignment marker 0x00 rate too low: {align_rate*100:.2f}% (< 99%)."
            )


def save_csv(csv_path: str, preset: str, diagnostics: dict) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("preset,packet_id_hex,count,mean_len,variance\n")
        for pkt_id in sorted(diagnostics["id_counts"].keys()):
            counts = diagnostics["id_counts"][pkt_id]
            lens = diagnostics["len_lists"][pkt_id]
            m = mean(lens)
            var = 0.0
            if len(lens) >= 2:
                mu = m
                var = mean([(x - mu) ** 2 for x in lens])
            f.write(f"{preset},0x{pkt_id:02x},{counts},{m:.2f},{var:.2f}\n")


def main():
    p = argparse.ArgumentParser(description="Stage A: Validate packet framing invariants")
    p.add_argument("--preset", required=True, help="Preset label (e.g., p1045)")
    p.add_argument(
        "--infile",
        required=False,
        help="Input .bin file path (default: data/{preset}/{preset}.bin)",
    )
    p.add_argument(
        "--csvout",
        required=False,
        help="CSV output path (default: data/{preset}/stageA_framing.csv)",
    )
    p.add_argument(
        "--non-strict",
        action="store_true",
        help="Relax alignment invariant to allow >=99% byte[13]==0x00",
    )
    args = p.parse_args()

    # Resolve default input file if not provided
    infile = args.infile or os.path.join("data", args.preset, f"{args.preset}.bin")
    if not os.path.isfile(infile):
        raise SystemExit(
            f"Input file not found: {infile}. Provide --infile or place a file at data/{args.preset}/{args.preset}.bin"
        )

    with open(infile, "rb") as f:
        blob = f.read()

    diags = analyze_packet_framing(blob)

    # Print summary
    print(f"Parsed {diags['total_packets']} packets from {infile}")
    print("Packet ID histogram (hex):")
    for pkt_id in sorted(diags["id_counts"].keys()):
        cnt = diags["id_counts"][pkt_id]
        lens = diags["len_lists"][pkt_id]
        m = mean(lens)
        print(f"  0x{pkt_id:02x}: count={cnt}, mean_len={m:.2f}")

    if diags["bad_id_counts"]:
        bad_str = ", ".join(
            f"0x{k:02x}={v}" for k, v in sorted(diags["bad_id_counts"].items())
        )
        print(f"Anomaly: Found impossible IDs -> {bad_str}")

    nonzero_align_total = sum(diags["align_nonzero"].values())
    if nonzero_align_total:
        detail = ", ".join(
            f"0x{k:02x}={v}" for k, v in sorted(diags["align_nonzero"].items())
        )
        print(
            f"Anomaly: byte[13] != 0x00 in {nonzero_align_total}/{diags['total_packets']} packets -> {detail}"
        )

    # Save CSV
    csv_path = args.csvout or os.path.join("data", args.preset, "stageA_framing.csv")
    save_csv(csv_path, args.preset, diags)
    print(f"Wrote CSV: {csv_path}")

    # Enforce invariants
    try:
        assert_invariants(diags, strict=not args.non_strict)
    except AssertionError as e:
        raise SystemExit(f"Stage A validation FAILED: {e}")

    print("Stage A validation PASSED.")


if __name__ == "__main__":
    main()


