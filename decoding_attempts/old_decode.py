# import struct
# from datetime import datetime, timezone
# from typing import Dict, Literal, Optional, Tuple
# from struct import unpack_from, error as StructError

# import numpy as np
# import pandas as pd

# # Packet structure -------------------
# # Offset (0-based)   Field
# # -----------------  -----------------------------------------
# # 0                  SUBPKT_LEN       (1 byte) [confirmed]
# # 1                  SUBPKT_N         (1 byte) [confirmed]
# # 2–5                SUBPKT_T         (uint32, ms since device start) [confirmed]
# # 6–8                SUBPKT_UNKNOWN1  (3 bytes, reserved?)
# # 9                  SUBPKT_ID        (freq/type nibbles) [confirmed]
# # 10–13              SUBPKT_METADATA  (4 bytes, little-endian; header metadata)
# # - interpretable as two little-endian uint16s:
# #   - u16_0 = bytes 10–11: high-variance 16-bit value (possibly per-packet offset / internal counter / fine-grained ID)
# #   - u16_1 = bytes 12–13: small discrete value ∈ {0,1,2,3} (likely a 2-bit slot/index / bank id)
# # - u8_3 (byte 13) is observed always 0 -> reserved/padding
# # 14...              SUBPKT_DATA      (multiplexed samples, tightly packed, repeating blocks)
# # - ACC/GYRO (TO BE CONFIRMED): Each block:
# #   - [tag byte: 0x47]
# #   - [4-byte block header (unknown; possibly sub-counter or timestamp offset)]
# #   - [N batched samples of 6 channels, interleaved per sample: (ACC_X, ACC_Y, ACC_Z, GYRO_X, GYRO_Y, GYRO_Z) x N]
# #   - [e.g., 36 bytes data: 18 signed 16-bit little-endian integers (<18h): 18 integers represent 6 channels x 3.
# #   - Multiple blocks per payload possible; search for all 0x47 tags to extract.
# # - Possible tags (TO BE CONFIRMED): 0x47 for ACCGYRO, 0x12 for EEG, 0x34 for optics, 0x98 for battery
# # - Other source (amused-py): 0xF4 for AGGYRO; 0xDB, 0xDF: EEG + PPG combined, 0xD9 for Mixed sensor data
# # Note: the payloads received might be concatenations of multiple subpackets (to be confirmed). Each subpacket starts with its own 1-byte length field (which includes the length byte itself), followed by the subpacket content.


# # ==============================================================================
# # Packet Info
# # ==============================================================================
# def extract_pkt_length(payload: bytes):
#     """
#     Extract the SUBPKT_LEN field (declared length) from a Muse payload.
#     """
#     if not payload or len(payload) < 14:  # minimum length for header
#         return None, False

#     declared_len = payload[0]
#     return declared_len, (declared_len == len(payload))


# def extract_pkt_n(payload: bytes):
#     """
#     Extract the SUBPKT_N field (1-byte sequence number) from a Muse payload.

#     - Located at offset 1 (0-based). Increments by 1 per packet, wraps at 255 -> 0.
#     - Useful for detecting dropped or out-of-order packets (quality check assessment).

#     Returns the integer sequence number (0-255).
#     """
#     return payload[1]


# def extract_pkt_time(payload: bytes):
#     """
#     Extract subpkt time from a single payload. 4-byte unsigned little-endian at offset 2 -> milliseconds.
#     """
#     # primary 4-byte little-endian at offset 2 (fixed from 3)
#     ms = struct.unpack_from("<I", payload, 2)[0]
#     return ms * 1e-3  # convert to seconds


# def extract_pkt_id(payload: bytes):
#     """
#     Extract and parse the ID byte from a Muse payload.
#     - ID byte is at offset 9 (0-based).
#     - Upper nibble = frequency code.
#     - Lower nibble = data type code.
#     Returns dict with raw codes and decoded labels.
#     """

#     # Lookup tables
#     FREQ_MAP = {
#         1: 256.0,
#         2: 128.0,
#         3: 64.0,
#         4: 52.0,
#         5: 32.0,
#         6: 16.0,
#         7: 10.0,
#         8: 1.0,
#         9: 0.1,
#     }
#     TYPE_MAP = {
#         1: "EEG4",
#         2: "EEG8",
#         3: "REF",
#         4: "Optics4",
#         5: "Optics8",
#         6: "Optics16",
#         7: "ACCGYRO",
#         8: "Battery",
#     }

#     id_byte = payload[9]
#     freq_code = (id_byte >> 4) & 0x0F
#     type_code = id_byte & 0x0F

#     return FREQ_MAP.get(freq_code), TYPE_MAP.get(type_code)


# def extract_pkt_metadata(payload: bytes):
#     """
#     Extract SUBPKT_METADATA from a Muse payload.

#     Structure (offsets are 0-based):
#       - bytes 10-11 : u16_0 (uint16 little-endian) - high-variance field (candidate: per-packet offset / internal counter)
#       - bytes 12-13 : u16_1 (uint16 little-endian) - low-variance small set {0,1,2,3} (candidate: slot/index/bank id)
#       - bytes 10..13 also viewed as u8_0..u8_3 for fine-grained inspection

#     It could be a slot_hint if equals raw_u16_1 if its value is in {0,1,2,3} else None.
#     """

#     # raw views
#     u16_0 = struct.unpack_from("<H", payload, 10)[0]
#     u16_1 = struct.unpack_from("<H", payload, 12)[0]
#     u8_0 = payload[10]
#     u8_1 = payload[11]
#     u8_2 = payload[12]
#     u8_3 = payload[13]

#     return {
#         "metadata_0": int(u16_0),
#         "metadata_1": int(u16_1),
#         # "u8_0": int(u8_0),
#         # "u8_1": int(u8_1),
#         # "u8_2": int(u8_2),
#         # "u8_3": int(u8_3),
#     }


# def extract_pkt_info(payload: bytes) -> Optional[Dict]:
#     pkt_len, valid = extract_pkt_length(payload)
#     if not valid or pkt_len is None:
#         return None
#     pkt_n = extract_pkt_n(payload)
#     pkt_time = extract_pkt_time(payload)
#     pkt_freq, pkt_type = extract_pkt_id(payload)
#     pkt_metadata = extract_pkt_metadata(payload)

#     return {
#         "length": pkt_len,
#         "sequence": pkt_n,
#         "time": pkt_time,
#         "frequency": pkt_freq,
#         "type": pkt_type,
#         "metadata_0": pkt_metadata["metadata_0"],
#         "metadata_1": pkt_metadata["metadata_1"],
#     }


# # ==============================================================================
# # Packet Data
# # ==============================================================================
# def extract_pkt_battery(payload: bytes, time):
#     """
#     Decode a Muse S (Athena) Battery packet.

#     Subpacket layout: battery payload (7 x uint16, little-endian).

#     Fields (based on MindMonitor / TI BQ fuel-gauge registers):
#       0: raw_state_of_charge (uint16): Reported as a fixed-point register (divide by 256 to get %)
#       1: raw_voltage (uint16): Fuel-gauge voltage reading (needs scaling by x16 to get millivolts)
#       2: raw_temperature (uint16): Temperature register. Unit unknown.
#       3-6: diagnostic / reserved registers (uint16 each): Exact purpose unclear (may be current, status flags, etc).

#     """
#     HEADER_LEN = 14
#     if len(payload) < HEADER_LEN + 14:
#         return None
#     data = payload[HEADER_LEN : HEADER_LEN + 14]
#     raw_soc, raw_mv, raw_temp, r1, r2, r3, r4 = struct.unpack("<7H", data)
#     return {
#         "battery": raw_soc / 256.0,
#         "voltage": raw_mv * 16,
#         "temperature": raw_temp,
#         "leftover": str(r1) + "," + str(r2) + "," + str(r3) + "," + str(r4),
#         "time": time,
#     }


# def extract_pkt_accgyro(payload: bytes, time: float) -> Optional[dict]:
#     """
#     Extract ACC/GYRO data from Muse payload.

#     FIX 1: Scan entire payload instead of skipping header (concatenated subpackets)
#     FIX 2: Use line timestamp directly instead of packet time + offsets
#     """
#     BLOCK_SIZE_BYTES = (
#         36  # 18 signed 16-bit values -> 36 bytes (3 samples x 6 channels)
#     )
#     N_SAMPLES_PER_BLOCK = 3
#     TAG = 0x47

#     # FIX 1: Don't skip header - scan entire payload
#     # Some payloads may be concatenated subpackets or have tags before byte 14
#     data = payload  # Changed from: data = payload[HEADER_LEN:]
#     L = len(data)

#     if L == 0:
#         return None

#     acc_samples = []
#     gyro_samples = []
#     ts = []

#     idx = 0
#     while idx < L:
#         # Find next tag position starting at idx
#         if data[idx] != TAG:
#             idx += 1
#             continue

#         # Found tag at idx; block payload starts after tag + 4-byte block header
#         start = idx + 1 + 4
#         end = start + BLOCK_SIZE_BYTES

#         # Not enough bytes for a full block; stop parsing this payload
#         if end > L:
#             break

#         block_bytes = data[start:end]

#         try:
#             vals = list(unpack_from("<18h", block_bytes, 0))
#         except StructError:
#             # Skip this block by advancing past the tag to avoid infinite loop
#             idx = idx + 1
#             continue

#         # Process N_SAMPLES_PER_BLOCK samples in this block
#         for s in range(N_SAMPLES_PER_BLOCK):
#             base = s * 6
#             acc_raw = vals[base : base + 3]
#             gyro_raw = vals[base + 3 : base + 6]

#             acc_scaled = [float(x) * 0.0000610352 for x in acc_raw]
#             gyro_scaled = [float(x) * -0.0074768 for x in gyro_raw]

#             acc_samples.append(acc_scaled)
#             gyro_samples.append(gyro_scaled)

#             # FIX 2: Use line timestamp directly like method 2
#             # (The cumulative offset calculation may be incorrect for concatenated packets)
#             ts.append(time)

#         # Advance index past this entire parsed block
#         idx = end

#     if not acc_samples and not gyro_samples:
#         return None

#     return {"time": ts, "acc": acc_samples, "gyro": gyro_samples}


# # def extract_pkt_accgyro(payload: bytes, time: float) -> Optional[dict]:
# #     """
# #     Tag-based ACC/GYRO parser (updated for Muse S format)

# #     Parsing rule:
# #       - After the 14-byte subpacket header the payload is scanned for tag 0xF4 (updated from 0x47).
# #       - Each tag is followed by a 4-byte block header.
# #       - The block payload starts at: tag_index + 1 + 4.
# #       - The block payload up to the next tag (or end-of-payload) is interpreted as
# #         tightly-packed samples where each sample = 9 bytes (6 signed 12-bit little-endian integers).
# #       - Only whole 9-byte samples are parsed from each block (trailing leftover bytes are ignored).
# #       - No continuous-stream fallback is used — if no valid tag-blocks with >=1 sample are found, returns None.

# #     Timing & scaling (unchanged):
# #       - Sample period = 1/52 s
# #       - ACC scaling = raw / 8192.0 -> g (assumes effective 16-bit after shift)
# #       - GYRO scaling = raw * 0.06103515625 -> dps (assumes effective 16-bit after shift)

# #     Returns:
# #       { "time": base_time,
# #         "acc": [(t, [ax,ay,az]), ...],
# #         "gyro": [(t, [gx,gy,gz]), ...] }
# #       or None if no valid samples parsed.
# #     """
# #     HEADER_LEN = 14
# #     SAMPLE_SIZE_BYTES = 9  # Updated: 9 bytes per sample (6 x 12-bit values)
# #     PERIOD = 1.0 / 52.0
# #     TAG = 0x47

# #     if len(payload) <= HEADER_LEN:
# #         return None

# #     data = payload[HEADER_LEN:]
# #     if not data:
# #         return None

# #     # Find candidate tag indices in the payload (relative to `data`)
# #     tag_idxs = [i for i, b in enumerate(data) if b == TAG]
# #     if not tag_idxs:
# #         return None

# #     acc_samples = []
# #     gyro_samples = []
# #     ts = []
# #     cumulative_sample_idx = 0  # Used to timestamp samples sequentially across blocks

# #     # Iterate through tags; use next tag position as boundary (or end of payload)
# #     for i, tag_idx in enumerate(tag_idxs):
# #         # Ensure 4-byte block header exists after tag
# #         if tag_idx + 1 + 4 > len(data):
# #             continue

# #         start = tag_idx + 1 + 4
# #         end = tag_idxs[i + 1] if i + 1 < len(tag_idxs) else len(data)
# #         if start >= end:
# #             continue

# #         block_bytes = data[start:end]
# #         n_samples = len(block_bytes) // SAMPLE_SIZE_BYTES
# #         if n_samples <= 0:
# #             continue

# #         # Process each 9-byte sample in the block
# #         for s in range(n_samples):
# #             sample_start = s * SAMPLE_SIZE_BYTES
# #             sample_bytes = block_bytes[sample_start : sample_start + SAMPLE_SIZE_BYTES]

# #             # Unpack 6 signed 12-bit values from 9 bytes (little-endian packed)
# #             raw_vals = []
# #             for k in range(
# #                 0, 9, 3
# #             ):  # Process in groups of 3 bytes (2 values per group)
# #                 b0, b1, b2 = sample_bytes[k : k + 3]
# #                 v1 = (b1 & 0x0F) << 8 | b0
# #                 if v1 & 0x800:  # Sign extend
# #                     v1 -= 0x1000
# #                 v2 = b2 << 4 | (b1 >> 4)
# #                 if v2 & 0x800:  # Sign extend
# #                     v2 -= 0x1000
# #                 raw_vals.extend([v1, v2])

# #             # Shift left by 4 to approximate original 16-bit resolution (Muse compresses to 12-bit for BLE efficiency)
# #             raw_vals = [v << 4 for v in raw_vals]

# #             # Split into ACC and GYRO, apply scaling
# #             acc_raw = raw_vals[0:3]
# #             gyro_raw = raw_vals[3:6]

# #             acc_samples.append([v / 8192.0 for v in acc_raw])
# #             gyro_samples.append([v * 0.06103515625 for v in gyro_raw])

# #             ts.append(time + cumulative_sample_idx * PERIOD)

# #             cumulative_sample_idx += 1

# #     if not acc_samples and not gyro_samples:
# #         return None

# #     return {"time": ts, "acc": acc_samples, "gyro": gyro_samples}


# # ==============================================================================
# # Main functions
# # ==============================================================================


# def decode_rawdata(lines: list[str]) -> Dict[str, pd.DataFrame]:
#     """
#     Decode raw lines into structured DataFrames.

#     Returns:
#         dict with keys:
#             "ACC"     -> DataFrame(time, ACC_X, ACC_Y, ACC_Z)
#             "GYRO"    -> DataFrame(time, GYRO_X, GYRO_Y, GYRO_Z)
#             "Battery" -> DataFrame(time, battery, temperature)
#             "Leftover"-> list of leftover strings (per-packet diagnostics)
#     """
#     decoded = []
#     leftovers_all = []

#     for ln in lines:
#         parts = ln.strip().split("\t")
#         if len(parts) < 3:
#             decoded.append(None)
#             continue

#         ts = datetime.fromisoformat(parts[0].replace("Z", "+00:00")).timestamp()
#         uuid = parts[1]
#         payload = bytes.fromhex(parts[2])

#         pkt_time = extract_pkt_time(payload)
#         pkt_type = extract_pkt_id(payload)[1]  # (freq, type)

#         pkt_decoded = None
#         if pkt_type == "ACCGYRO":
#             pkt_decoded = extract_pkt_accgyro(payload, time=pkt_time)
#         elif pkt_type == "Battery":
#             pkt_decoded = extract_pkt_battery(payload, time=pkt_time)

#         decoded.append(
#             {
#                 "uuid": uuid,
#                 "line_time": ts,
#                 "pkt_time": pkt_time,
#                 "pkt_type": pkt_type,
#                 "data": pkt_decoded,
#             }
#         )

#     # Collect data -----------------------------------------------------
#     acc, gyro, batt = [], [], []

#     for pkt in decoded:
#         if not pkt or not pkt["data"]:
#             continue

#         data = pkt["data"]
#         if pkt["pkt_type"] == "ACCGYRO":

#             # expand acc + gyro
#             for i, _ in enumerate(data["acc"]):
#                 acc.append([data["time"][i], *data["acc"][i]])
#                 gyro.append([data["time"][i], *data["gyro"][i]])

#         elif pkt["pkt_type"] == "Battery":
#             batt.append([data["time"], data["battery"], data["temperature"]])

#     # Convert to DataFrames --------------------------------------------
#     df_acc = (
#         pd.DataFrame(acc, columns=["time", "ACC_X", "ACC_Y", "ACC_Z"])
#         if acc
#         else pd.DataFrame(columns=["time", "ACC_X", "ACC_Y", "ACC_Z"])
#     )

#     df_gyro = (
#         pd.DataFrame(gyro, columns=["time", "GYRO_X", "GYRO_Y", "GYRO_Z"])
#         if gyro
#         else pd.DataFrame(columns=["time", "GYRO_X", "GYRO_Y", "GYRO_Z"])
#     )

#     df_batt = (
#         pd.DataFrame(batt, columns=["time", "battery", "temperature"])
#         if batt
#         else pd.DataFrame(columns=["time", "battery", "temperature"])
#     )

#     return {
#         "ACC": df_acc,
#         "GYRO": df_gyro,
#         "Battery": df_batt,
#         "Leftover": leftovers_all,
#         "Raw": decoded,
#     }


# def decode_rawdata_fixed(lines: list[str]) -> Dict[str, pd.DataFrame]:
#     """
#     Decode raw lines into structured DataFrames.

#     KEY FIX: Scan ALL packets for ACC/GYRO data, not just those with type="ACCGYRO"
#     Some packets may contain 0x47 tags even if the ID byte says otherwise.

#     Returns:
#         dict with keys:
#             "ACC"     -> DataFrame(time, ACC_X, ACC_Y, ACC_Z)
#             "GYRO"    -> DataFrame(time, GYRO_X, GYRO_Y, GYRO_Z)
#             "Battery" -> DataFrame(time, battery, temperature)
#     """
#     decoded = []

#     for ln in lines:
#         parts = ln.strip().split("\t")
#         if len(parts) < 3:
#             decoded.append(None)
#             continue

#         ts = datetime.fromisoformat(parts[0].replace("Z", "+00:00")).timestamp()
#         uuid = parts[1]
#         payload = bytes.fromhex(parts[2])

#         pkt_time = extract_pkt_time(payload)
#         pkt_type = extract_pkt_id(payload)[1] if len(payload) > 9 else None

#         pkt_decoded = None

#         # FIX: Always try to extract ACC/GYRO data regardless of declared packet type
#         # Method 2 scans all packets, so we should too
#         pkt_decoded_accgyro = extract_pkt_accgyro(payload, time=pkt_time)

#         # Also check for battery data if packet type indicates it
#         pkt_decoded_battery = None
#         if pkt_type == "Battery":
#             pkt_decoded_battery = extract_pkt_battery(payload, time=pkt_time)

#         decoded.append(
#             {
#                 "uuid": uuid,
#                 "line_time": ts,
#                 "pkt_time": pkt_time,
#                 "pkt_type": pkt_type,
#                 "data_accgyro": pkt_decoded_accgyro,
#                 "data_battery": pkt_decoded_battery,
#             }
#         )

#     # Collect data -----------------------------------------------------
#     acc, gyro, batt = [], [], []

#     for pkt in decoded:
#         if not pkt:
#             continue

#         # Process ACC/GYRO data
#         if pkt["data_accgyro"]:
#             data = pkt["data_accgyro"]
#             for i in range(len(data["acc"])):
#                 acc.append([data["time"][i], *data["acc"][i]])
#                 gyro.append([data["time"][i], *data["gyro"][i]])

#         # Process Battery data
#         if pkt["data_battery"]:
#             data = pkt["data_battery"]
#             batt.append([data["time"], data["battery"], data["temperature"]])

#     # Convert to DataFrames --------------------------------------------
#     df_acc = (
#         pd.DataFrame(acc, columns=["time", "ACC_X", "ACC_Y", "ACC_Z"])
#         if acc
#         else pd.DataFrame(columns=["time", "ACC_X", "ACC_Y", "ACC_Z"])
#     )

#     df_gyro = (
#         pd.DataFrame(gyro, columns=["time", "GYRO_X", "GYRO_Y", "GYRO_Z"])
#         if gyro
#         else pd.DataFrame(columns=["time", "GYRO_X", "GYRO_Y", "GYRO_Z"])
#     )

#     df_batt = (
#         pd.DataFrame(batt, columns=["time", "battery", "temperature"])
#         if batt
#         else pd.DataFrame(columns=["time", "battery", "temperature"])
#     )

#     return {
#         "ACC": df_acc,
#         "GYRO": df_gyro,
#         "Battery": df_batt,
#         "Raw": decoded,
#     }


# # ==============================================================================
# # AthenaParse
# # ==============================================================================
# import re
# import struct
# from typing import List, Dict, Tuple, Optional
# import pandas as pd
# from datetime import datetime

# # -------------------------------------------------------------------------
# # Utilities
# # -------------------------------------------------------------------------


# def bytes_to_bitarray_lsb_first(data: bytes) -> List[int]:
#     bits: List[int] = []
#     for b in data:
#         for i in range(8):
#             bits.append((b >> i) & 1)
#     return bits


# def parse_uint14_le_values(buf: bytes) -> List[int]:
#     bits = bytes_to_bitarray_lsb_first(buf)
#     width = 14
#     n_vals = len(bits) // width
#     out: List[int] = []
#     for i in range(n_vals):
#         chunk = bits[i * width : (i + 1) * width]
#         val = 0
#         for bit_index, bit in enumerate(chunk):
#             if bit:
#                 val |= 1 << bit_index
#         out.append(val)
#     return out


# # -------------------------------------------------------------------------
# # Packet parser
# # -------------------------------------------------------------------------


# def parse_packet(data: bytes, tag: int, tag_index: int):
#     payload_start = tag_index + 1 + 4

#     if tag == 0x12:  # EEG
#         payload_len = 28
#         end_index = payload_start + payload_len
#         if end_index > len(data):
#             return end_index, None, []
#         block28 = data[payload_start:end_index]
#         values = parse_uint14_le_values(block28)
#         scaled = [v * (1450 / 16383) for v in values]
#         return (
#             end_index,
#             "EEG",
#             [{"EEG_ch{}".format(i): float(v) for i, v in enumerate(scaled)}],
#         )

#     elif tag == 0x47:  # ACC + GYRO
#         ints_needed = 18
#         bytes_needed = ints_needed * 2
#         end_index = payload_start + bytes_needed
#         if end_index > len(data):
#             return end_index, None, []
#         block = data[payload_start:end_index]
#         vals = list(struct.unpack("<18h", block))
#         out = []
#         for i in range(3):  # 3 samples
#             base = i * 6
#             acc_raw = vals[base : base + 3]
#             gyro_raw = vals[base + 3 : base + 6]
#             acc_scaled = [float(x) * 0.0000610352 for x in acc_raw]
#             gyro_scaled = [float(x) * -0.0074768 for x in gyro_raw]
#             out.append(
#                 {
#                     "ACC_X": acc_scaled[0],
#                     "ACC_Y": acc_scaled[1],
#                     "ACC_Z": acc_scaled[2],
#                     "GYRO_X": gyro_scaled[0],
#                     "GYRO_Y": gyro_scaled[1],
#                     "GYRO_Z": gyro_scaled[2],
#                 }
#             )
#         return end_index, "ACC_GYRO", out

#     else:
#         return tag_index + 1, None, []


# # -------------------------------------------------------------------------
# # Main decoder
# # -------------------------------------------------------------------------


# def decode_rawdata2(lines: List[str]) -> Dict[str, pd.DataFrame]:
#     """
#     Decode raw Muse log lines into DataFrames for each modality.
#     Each line is 'timestamp \t uuid \t hexstring'.
#     """
#     dfs: Dict[str, List[dict]] = {
#         "EEG": [],
#         "ACC": [],
#         "GYRO": [],
#         "BATTERY": [],
#         "OPTICAL": [],
#     }

#     for line in lines:
#         try:
#             ts_str, uuid, hexstr = line.split("\t")
#         except ValueError:
#             continue

#         try:
#             ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
#         except Exception:
#             ts = None

#         try:
#             data = bytes.fromhex(hexstr)
#         except ValueError:
#             continue

#         idx = 0
#         while idx < len(data):
#             tag = data[idx]
#             next_idx, name, entries = parse_packet(data, tag, idx)
#             if name == "EEG":
#                 for e in entries:
#                     e["time"] = ts
#                     dfs["EEG"].append(e)
#             elif name == "ACC_GYRO":
#                 for e in entries:
#                     e["time"] = ts
#                     dfs["ACC"].append(
#                         {
#                             "time": ts,
#                             "ACC_X": e["ACC_X"],
#                             "ACC_Y": e["ACC_Y"],
#                             "ACC_Z": e["ACC_Z"],
#                         }
#                     )
#                     dfs["GYRO"].append(
#                         {
#                             "time": ts,
#                             "GYRO_X": e["GYRO_X"],
#                             "GYRO_Y": e["GYRO_Y"],
#                             "GYRO_Z": e["GYRO_Z"],
#                         }
#                     )
#             idx = max(next_idx, idx + 1)

#     out = {}
#     for k, v in dfs.items():
#         if v:
#             out[k] = pd.DataFrame(v).sort_values("time")
#     return out


# # ==============================================================================
# #  Test
# # ==============================================================================
# import urllib.request

# import matplotlib.pyplot as plt

# url = f"https://raw.githubusercontent.com/DominiqueMakowski/MuseLSL3/refs/heads/main/decoding_attempts/data_raw/data_p1045.txt"
# lines = urllib.request.urlopen(url).read().decode("utf-8").splitlines()
# data = decode_rawdata_fixed(lines)
# data["ACC"].plot(x="time", y=["ACC_X", "ACC_Y", "ACC_Z"], subplots=True)
# data2 = decode_rawdata2(lines)
# data2["ACC"].plot(x="time", y=["ACC_X", "ACC_Y", "ACC_Z"], subplots=True)

# # import MuseLSL3

# for preset in ["p21", "p1034", "p1045"]:
#     preset = "p1045"

#     print(f"Decoding preset {preset}")
#     url = f"https://raw.githubusercontent.com/DominiqueMakowski/MuseLSL3/refs/heads/main/decoding_attempts/data_raw/data_{preset}.txt"

#     lines = urllib.request.urlopen(url).read().decode("utf-8").splitlines()

#     data = decode_rawdata(lines)

#     data["ACC"].plot(x="time", y=["ACC_X", "ACC_Y", "ACC_Z"], subplots=True)
#     data["GYRO"].plot(x="time", y=["GYRO_X", "GYRO_Y", "GYRO_Z"], subplots=True)

#     if len(data["ACC"]) == 0 or len(data["GYRO"]) == 0:
#         print("No data decoded")
#         continue
#     len(data["ACC"]) / 60
#     data["ACC"].plot(x="time", y=["ACC_X", "ACC_Y", "ACC_Z"], subplots=True)
#     data["GYRO"].plot(x="time", y=["GYRO_X", "GYRO_Y", "GYRO_Z"], subplots=True)
#     data["ACC"].plot(y="time")  # Ok, monotonic
#     data["Leftover"]

#     # # Battery
#     # print(f"Battery data: {len(data['Battery'])} samples")
#     # if len(data["Battery"]) > 0:
#     #     print(data["Battery"].describe())
#     #     data["Battery"].plot(x="time", y="battery")
#     #     data["Battery"].plot(x="time", y="temperature")

#     # ACC + GYRO
#     # Print proportion of empty data
#     accgyro_pkts = []
#     for pkt in data["Raw"]:
#         if pkt and pkt["pkt_type"] == "ACCGYRO":
#             accgyro_pkts.append(pkt)
#     n_accgyro = len(accgyro_pkts)
#     n_accgyro / 60
#     prop_none = sum(1 for pkt in accgyro_pkts if pkt["data"] is None) / len(
#         accgyro_pkts
#     )
#     prop_none

#     # print(
#     #     f"Same length: {len(data['ACC']) == len(data['GYRO'])}"
#     # )  # Both have same length, good


# # The method2 returns more ACCGYRO samples (e.g., 3000+ rows) while method1 only 800, and the signal looks also a bit more like ACCGYRO (although with some noise). This makes me think that method2 does a better job than method1.

# # However, method2 seems to assume that ACCGYRO samples can be found in any packet, not just packets with the type being accgyro.

# # I am attaching a bluetooth HCI snoop (analyzed with Wireshark) from a decompiled MindMonitor apk. Can you analize it and tell me if it is useful to validate the fact that ACCGYRO can be present in any packet, not just accgyro packet types?

# # 1. Using these decoding functions on a file, I get less acc/gyro samples than the 52 Hz. The signals also do not look like smooth ACC/GYRO traces. This suggests that something is wrong with the decoding.
# # 2. It seems like they are two approaches for decoding ACCGYRO: a tag-based parsing approach versus a continuous stream parsing approach. The tag-based method operates on the assumption that specific byte values (e.g., 0x47) are embedded in the data to mark the beginning of distinct data blocks. A parser using this strategy scans the payload for these predefined tags and begins decoding at their location. However, this approach might fail if the tag's value coincidentally appears as part of the actual sample data, as the parser will incorrectly identify a new block. The continuous stream approach treats the entire data payload as a single, uninterrupted sequence of tightly packed samples. Knowing the fixed size and structure of each sample, this parser simply reads the stream sequentially, slicing it into expected sized chunks. This method interprets every byte as part of a sample.
# # 3. I am attaching an Android debug log from a custom decompiled MindMonitor apk and a bluetooth HCI snoop (analyzed with Wireshark). Can we use this information to find what which approach is correct. If the tag-based approach is correct, is this tag always 0x47? How to handle cases where the tag appears in the data? If the stream approach is better, are we sure we know the correct sample structure, since it seems like there variable sample lengths? Could we make use of the Metadata info to for instance predict the number of samples?
# # 4. To help, I am also attaching a sample of raw data recording, and pasting a decoding script that claims to have cracked the decoding. https://github.com/AbosaSzakal/MuseAthenaDataformatParser/blob/main/athena_packet_decoder.py

# # 3. Compare  that against solutions in https://github.com/AbosaSzakal/MuseAthenaDataformatParser and https://github.com/Amused-EEG/amused-py

# # The ACC/GYRO signals look bad however, with both are likely contaminated and not pure and correctly decoded accgyro data.
# # 1. Compare the ACCGYRO decoding functions and outline the differences and similarities.

# # 4. Write me a drop-in replacement function that implements the correct decoding method.
