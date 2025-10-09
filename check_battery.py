"""Quick script to examine battery readings in test_accgyro.txt"""

import struct
from MuseLSL3.decode import parse_message

# Compare battery readings from different files
test_files = {
    "test_accgyro.txt": "tests/test_data/test_accgyro.txt",
    "test_battery_16_80.txt": "tests/test_data/test_battery_16_80.txt",
    "test_battery_90_40.txt": "tests/test_data/test_battery_90_40.txt",
    "test_battery_58_27.txt": "tests/test_data/test_battery_58_27.txt",
}

for filename, filepath in test_files.items():
    print(f"\n{'='*60}")
    print(f"File: {filename}")
    print("=" * 60)

    with open(filepath, "r", encoding="utf-8") as f:
        messages = f.readlines()

    battery = []
    for m in messages:
        subpackets = parse_message(m)
        for sp in subpackets:
            if sp["pkt_type"] == "Battery" and sp["data_battery"]:
                bat_array = sp["data_battery"][0]
                battery.append(
                    {
                        "pkt_time": sp["pkt_time"],
                        "battery_percent": bat_array[1],
                        "message_time": sp["message_time"],
                        "pkt_offset": sp["pkt_offset"],
                    }
                )

    print(f"Found {len(battery)} battery readings\n")

    if battery:
        print("First 5 battery readings:")
        for i, b in enumerate(battery[:5]):
            print(
                f"  {i+1}. Time: {b['pkt_time']:.3f}s, Battery: {b['battery_percent']:.2f}%"
            )

        print("\nRaw data inspection for first battery packet:")
        # Parse the first message with a battery packet
        for m in messages:
            subpackets = parse_message(m)
            for sp in subpackets:
                if sp["pkt_type"] == "Battery" and sp["data_battery"]:
                    # Get the raw packet to examine
                    payload = bytes.fromhex(m.strip().split("\t")[2])
                    pkt = payload[sp["pkt_offset"] : sp["pkt_offset"] + sp["pkt_len"]]
                    data_section = pkt[14:]

                    print(f"  Packet length: {len(pkt)} bytes")
                    print(f"  Data section: {len(data_section)} bytes")
                    print(f"  First 20 bytes (hex): {data_section[:20].hex()}")
                    print(f"  First 14 bytes breakdown:")

                    # Parse first 14 bytes
                    raw_soc = struct.unpack("<H", data_section[0:2])[0]
                    voltage = struct.unpack("<H", data_section[2:4])[0]
                    temp = struct.unpack("<H", data_section[4:6])[0]
                    field1 = struct.unpack("<H", data_section[6:8])[0]
                    field2 = struct.unpack("<H", data_section[8:10])[0]
                    field3 = struct.unpack("<H", data_section[10:12])[0]
                    field4 = struct.unpack("<H", data_section[12:14])[0]

                    print(
                        f"    Bytes 0-1 (raw_soc): {raw_soc} (0x{raw_soc:04x}) -> {raw_soc/256.0:.2f}%"
                    )
                    print(f"    Bytes 2-3 (voltage): {voltage} (0x{voltage:04x})")
                    print(f"    Bytes 4-5 (temp): {temp} (0x{temp:04x})")
                    print(f"    Bytes 6-7: {field1} (0x{field1:04x})")
                    print(f"    Bytes 8-9: {field2} (0x{field2:04x})")
                    print(f"    Bytes 10-11: {field3} (0x{field3:04x})")
                    print(f"    Bytes 12-13: {field4} (0x{field4:04x})")

                    # Exit after first battery packet
                    break
            else:
                continue
            break
