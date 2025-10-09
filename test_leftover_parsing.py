"""Test the updated decoder with leftover parsing"""

from MuseLSL3.decode import parse_message
import numpy as np

# Read test data
with open("tests/test_data/test_accgyro.txt", "r", encoding="utf-8") as f:
    messages = f.readlines()[:100]

print("=" * 80)
print("TESTING UPDATED DECODER - Parsing leftovers for additional sensor data")
print("=" * 80)

# Count primary vs leftover data
primary_accgyro = 0
leftover_accgyro = 0
primary_battery = 0
leftover_battery = 0

total_packets = 0

for msg_idx, message in enumerate(messages):
    subpackets = parse_message(message)

    for sp in subpackets:
        if not sp["pkt_valid"]:
            continue

        total_packets += 1

        # Count ACCGYRO data
        if sp["pkt_type"] == "ACCGYRO" and len(sp["data_accgyro"]) > 0:
            primary_accgyro += 1
            if len(sp["data_accgyro"]) > 1:
                leftover_accgyro += len(sp["data_accgyro"]) - 1

        # Count Battery data
        if sp["pkt_type"] == "Battery" and len(sp["data_battery"]) > 0:
            primary_battery += 1
            if len(sp["data_battery"]) > 1:
                leftover_battery += len(sp["data_battery"]) - 1

print(f"\nAnalyzed {len(messages)} messages, {total_packets} packets")

print(f"\nACCGYRO data:")
print(f"  Primary packets with ACCGYRO: {primary_accgyro}")
print(f"  Additional ACCGYRO from leftovers: {leftover_accgyro}")
print(f"  Total ACCGYRO data arrays: {primary_accgyro + leftover_accgyro}")

print(f"\nBattery data:")
print(f"  Primary packets with Battery: {primary_battery}")
print(f"  Additional Battery from leftovers: {leftover_battery}")
print(f"  Total Battery data arrays: {primary_battery + leftover_battery}")

# Show a few examples
print(f"\n{'='*80}")
print("EXAMPLES OF PACKETS WITH LEFTOVER DATA:")
print("=" * 80)

examples_shown = 0
for msg_idx, message in enumerate(messages):
    subpackets = parse_message(message)

    for sp in subpackets:
        if sp["pkt_valid"] and len(sp["data_accgyro"]) > 1:
            examples_shown += 1
            print(
                f"\nMessage {msg_idx}, Primary packet: {sp['pkt_type']}, counter={sp['pkt_n']}"
            )
            print(
                f"  Found {len(sp['data_accgyro'])} ACCGYRO data arrays (1 primary + {len(sp['data_accgyro'])-1} from leftovers)"
            )

            # Show shape of each array
            for i, arr in enumerate(sp["data_accgyro"]):
                print(
                    f"    Array {i+1}: shape {arr.shape}, time range: {arr[0,0]:.3f} - {arr[-1,0]:.3f}s"
                )

            if examples_shown >= 5:
                break

    if examples_shown >= 5:
        break

# Check Battery leftovers
print(f"\n{'='*80}")
print("BATTERY DATA FROM LEFTOVERS:")
print("=" * 80)

battery_examples = 0
for msg_idx, message in enumerate(messages):
    subpackets = parse_message(message)

    for sp in subpackets:
        if sp["pkt_valid"] and len(sp["data_battery"]) > 1:
            battery_examples += 1
            print(
                f"\nMessage {msg_idx}, Primary packet: {sp['pkt_type']}, counter={sp['pkt_n']}"
            )
            print(
                f"  Found {len(sp['data_battery'])} Battery readings (1 primary + {len(sp['data_battery'])-1} from leftovers)"
            )

            for i, arr in enumerate(sp["data_battery"]):
                print(f"    Reading {i+1}: time={arr[0]:.3f}s, battery={arr[1]:.2f}%")

            if battery_examples >= 3:
                break

    if battery_examples >= 3:
        break

if battery_examples == 0:
    print("  (No battery data found in leftovers in this sample)")

print(f"\n{'='*80}")
print("CONCLUSION")
print("=" * 80)

if leftover_accgyro > 0 or leftover_battery > 0:
    print(
        "✓✓✓ SUCCESS! Decoder is now extracting additional sensor data from leftovers!"
    )
    print(f"✓ Found {leftover_accgyro} additional ACCGYRO data arrays")
    print(f"✓ Found {leftover_battery} additional Battery readings")
else:
    print("⚠ No additional data found in leftovers")
    print("  This could mean:")
    print("  - Leftovers don't contain TAG+data structure in this sample")
    print("  - Or the structure is different than expected")
