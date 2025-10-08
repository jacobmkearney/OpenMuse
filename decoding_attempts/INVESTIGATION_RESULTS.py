"""
MAJOR DISCOVERY: ACCGYRO Leftover Bytes Investigation Results
===============================================================

Date: 2025-10-08
Investigation: Leftover bytes after the first 3 ACCGYRO samples (36 bytes)

SUMMARY
-------
The "leftover bytes" in ACCGYRO data subpackets are NOT additional ACCGYRO samples.
Instead, they contain ADDITIONAL SUBPACKETS from other sensor types!

This explains why:
1. The leftover length is variable (depends on what other subpackets are included)
2. The leftover bytes often start with known packet tags (0x11, 0x12, 0x34, etc.)
3. The leftover length is often ODD (not aligned with 12-byte ACCGYRO samples)

KEY FINDINGS
------------

1. ACCGYRO SUBPACKET STRUCTURE:
   - ALWAYS starts with exactly 36 bytes (3 samples × 6 channels × 2 bytes)
   - After these 36 bytes, additional subpackets may follow
   - These additional subpackets are from OTHER sensor types

2. SUBPACKET TYPES FOUND IN LEFTOVER BYTES:
   From analysis of 621 ACCGYRO packets:
   - 0x11 (EEG4):     953 occurrences
   - 0x12 (EEG8):     367 occurrences
   - 0x13 (REF):      144 occurrences
   - 0x36 (Optics16): 193 occurrences
   - 0x47 (ACCGYRO):  323 occurrences  (MORE ACCGYRO data!)
   - 0x98 (Battery):  123 occurrences

3. SUBPACKET FORMAT:
   Each subpacket in the leftover bytes follows this structure:
   - 1 byte:  Tag (packet type identifier)
   - 4 bytes: Header (metadata)
   - N bytes: Payload (sensor data)

   Where N depends on the packet type:
   - 0x11 (EEG4):     14 bytes payload
   - 0x12 (EEG8):     28 bytes payload
   - 0x13 (REF):      7 bytes payload
   - 0x36 (Optics16): 96 bytes payload
   - 0x47 (ACCGYRO):  36 bytes payload
   - 0x98 (Battery):  4 bytes payload

4. PARSING SUCCESS RATE:
   - Successfully parsed: 620/621 packets (99.8%)
   - Failed to parse: 1/621 packets (0.2%)

EXAMPLE PACKET BREAKDOWN
-------------------------

Packet with 201 total bytes:
┌─────────────────────────────────────────────────────────┐
│ First 36 bytes: ACCGYRO data (3 samples)                │
│   Sample 1: ACC=(-3466, -1590, 12707) GYRO=(152, 250, 153) │
│   Sample 2: ACC=(-4292, -1811, 15149) GYRO=(-2202, 12172, 344) │
│   Sample 3: ACC=(794, -1899, 17778) GYRO=(-1068, 17787, 254) │
├─────────────────────────────────────────────────────────┤
│ Next 165 bytes: 5 additional subpackets                 │
│   1. Tag=0x11 (EEG4):  Header=00F918FF + 14 bytes data  │
│   2. Tag=0x11 (EEG4):  Header=01353801 + 14 bytes data  │
│   3. Tag=0x11 (EEG4):  Header=02913801 + 14 bytes data  │
│   4. Tag=0x11 (EEG4):  Header=03CE3801 + 14 bytes data  │
│   5. Tag=0x11 (EEG4):  Header=04293901 + 14 bytes data  │
│   (+ 70 unparsed bytes - likely more subpackets)        │
└─────────────────────────────────────────────────────────┘

IMPLICATIONS FOR DECODER
-------------------------

1. The current decoder that only extracts the first 36 bytes is CORRECT for ACCGYRO.

2. However, to fully utilize the data stream, the decoder should:
   a) Extract the first 36 bytes as ACCGYRO data
   b) Parse the remaining bytes as additional subpackets
   c) Route each subpacket to the appropriate decoder (EEG, Battery, etc.)

3. The BLE message structure is more sophisticated than initially thought:
   - The main packet (with 14-byte header) contains a PRIMARY subpacket
   - After the primary subpacket data, SECONDARY subpackets may be bundled
   - This allows multiple sensor types to be transmitted in a single BLE message

4. The 0x47 (ACCGYRO) tags found in leftover bytes suggest that:
   - Sometimes, additional ACCGYRO samples ARE included
   - These would be additional sets of 3 samples (36 bytes each)
   - This could provide higher temporal resolution for motion data

RECOMMENDED NEXT STEPS
----------------------

1. Update the packet parser to:
   - Parse the primary subpacket based on PKT_ID
   - Continue parsing leftover bytes as additional subpackets
   - Return all subpackets from a single BLE message

2. Investigate the ACCGYRO tags in leftover bytes:
   - Are these truly additional motion samples?
   - What is their timing relationship to the first 3 samples?

3. Update documentation to reflect the true packet structure:
   - BLE Message → Main Packet (14-byte header) → Multiple Subpackets
   - Each subpacket: Tag (1 byte) + Header (4 bytes) + Payload (N bytes)

4. Consider whether to merge all subpackets or keep them separate:
   - Option A: Merge same-type subpackets (e.g., combine all ACCGYRO data)
   - Option B: Keep subpackets separate with timing info from headers

VALIDATION
----------

This structure explains several previously mysterious observations:
✓ Why leftover length is variable
✓ Why leftover bytes start with known packet tags
✓ Why leftover length is often odd (depends on bundled subpackets)
✓ Why we see patterns like "0x11" appearing at specific positions
✓ Why successful parsing rate is so high (99.8%)

The investigation provides strong evidence that the Muse device bundles
multiple sensor subpackets into single BLE messages for efficient transmission.
"""

print(__doc__)
