"""
Unit tests for the decode_new module.

Tests the parse_message() and decode_battery() functions with real BLE message data from Muse devices.
"""

import unittest
import os
import struct
import numpy as np
from datetime import datetime
from MuseLSL3.decode_new import parse_message, decode_battery, FREQ_MAP, TYPE_MAP


class TestParseMessage(unittest.TestCase):
    """Test suite for parse_message() function."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures - locate test data files."""
        cls.test_data_dir = os.path.join(os.path.dirname(__file__), "test_data")

        # All 15 presets for comprehensive testing
        cls.test_files = {
            "p20": os.path.join(cls.test_data_dir, "data_p20.txt"),
            "p21": os.path.join(cls.test_data_dir, "data_p21.txt"),
            "p50": os.path.join(cls.test_data_dir, "data_p50.txt"),
            "p51": os.path.join(cls.test_data_dir, "data_p51.txt"),
            "p60": os.path.join(cls.test_data_dir, "data_p60.txt"),
            "p61": os.path.join(cls.test_data_dir, "data_p61.txt"),
            "p1034": os.path.join(cls.test_data_dir, "data_p1034.txt"),
            "p1035": os.path.join(cls.test_data_dir, "data_p1035.txt"),
            "p1041": os.path.join(cls.test_data_dir, "data_p1041.txt"),
            "p1042": os.path.join(cls.test_data_dir, "data_p1042.txt"),
            "p1043": os.path.join(cls.test_data_dir, "data_p1043.txt"),
            "p1044": os.path.join(cls.test_data_dir, "data_p1044.txt"),
            "p1045": os.path.join(cls.test_data_dir, "data_p1045.txt"),
            "p1046": os.path.join(cls.test_data_dir, "data_p1046.txt"),
            "p4129": os.path.join(cls.test_data_dir, "data_p4129.txt"),
        }

        # Verify test data files exist
        for preset, filepath in cls.test_files.items():
            if not os.path.exists(filepath):
                raise FileNotFoundError(
                    f"Test data file not found: {filepath}\n"
                    f"Please ensure test data files are in {cls.test_data_dir}"
                )

    def test_basic_parsing(self):
        """Test basic parsing of a single message."""
        # Read one message from p20
        with open(self.test_files["p20"], "r", encoding="utf-8") as f:
            message = f.readline()

        subpackets = parse_message(message)

        # Should return a list
        self.assertIsInstance(subpackets, list)
        self.assertGreater(len(subpackets), 0)

        # Check first subpacket has all required fields
        subpkt = subpackets[0]
        required_fields = {
            "message_time",
            "message_uuid",
            "pkt_offset",
            "pkt_len",
            "pkt_n",
            "pkt_time",
            "pkt_unknown1",
            "pkt_freq",
            "pkt_type",
            "pkt_unknown2",
            "pkt_data",
            "pkt_valid",
        }
        self.assertEqual(set(subpkt.keys()), required_fields)

    def test_field_types(self):
        """Test that all fields have correct types."""
        with open(self.test_files["p20"], "r", encoding="utf-8") as f:
            message = f.readline()

        subpackets = parse_message(message)
        subpkt = subpackets[0]

        # Type checks
        self.assertIsInstance(subpkt["message_time"], datetime)
        self.assertIsInstance(subpkt["message_uuid"], str)
        self.assertIsInstance(subpkt["pkt_offset"], int)
        self.assertIsInstance(subpkt["pkt_len"], int)
        self.assertIsInstance(subpkt["pkt_n"], int)
        self.assertIsInstance(subpkt["pkt_time"], float)
        self.assertIsInstance(subpkt["pkt_unknown1"], bytes)
        self.assertIn(type(subpkt["pkt_freq"]), [float, type(None)])
        self.assertIn(type(subpkt["pkt_type"]), [str, type(None)])
        self.assertIsInstance(subpkt["pkt_unknown2"], bytes)
        self.assertIsInstance(subpkt["pkt_valid"], bool)
        # pkt_data can be bytes (raw) or tuple (decoded)
        self.assertIn(type(subpkt["pkt_data"]), [bytes, tuple])

    def test_field_constraints(self):
        """Test that fields meet expected constraints."""
        with open(self.test_files["p20"], "r", encoding="utf-8") as f:
            message = f.readline()

        subpackets = parse_message(message)
        subpkt = subpackets[0]

        # Offset should be 0 for first packet
        self.assertEqual(subpkt["pkt_offset"], 0)

        # Length should be >= 14 (minimum header)
        self.assertGreaterEqual(subpkt["pkt_len"], 14)

        # Counter should be 0-255
        self.assertGreaterEqual(subpkt["pkt_n"], 0)
        self.assertLessEqual(subpkt["pkt_n"], 255)

        # Unknown1 should be 3 bytes
        self.assertEqual(len(subpkt["pkt_unknown1"]), 3)

        # Unknown2 should be 3 bytes
        self.assertEqual(len(subpkt["pkt_unknown2"]), 3)

        # If valid, frequency and type should be recognized
        if subpkt["pkt_valid"]:
            self.assertIsNotNone(subpkt["pkt_freq"])
            self.assertIsNotNone(subpkt["pkt_type"])
            self.assertIn(subpkt["pkt_freq"], FREQ_MAP.values())
            self.assertIn(subpkt["pkt_type"], TYPE_MAP.values())

    def test_100_percent_coverage(self):
        """Test that 100% of payload bytes are decoded (no leftovers)."""
        for preset, filepath in self.test_files.items():
            with self.subTest(preset=preset):
                with open(filepath, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                for line_num, message in enumerate(lines[:10]):  # Test first 10
                    if not message.strip():
                        continue

                    # Extract payload size
                    parts = message.strip().split("\t", 2)
                    payload = bytes.fromhex(parts[2].strip())
                    payload_size = len(payload)

                    # Parse message
                    subpackets = parse_message(message)

                    # Calculate decoded bytes
                    decoded_bytes = sum(s["pkt_len"] for s in subpackets)

                    # Should decode 100%
                    self.assertEqual(
                        decoded_bytes,
                        payload_size,
                        f"Line {line_num+1}: Decoded {decoded_bytes}/{payload_size} bytes",
                    )

    def test_100_percent_validity(self):
        """Test that 100% of parsed subpackets are valid."""
        for preset, filepath in self.test_files.items():
            with self.subTest(preset=preset):
                with open(filepath, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                total_subpackets = 0
                valid_subpackets = 0

                for message in lines[:10]:  # Test first 10
                    if not message.strip():
                        continue

                    subpackets = parse_message(message)
                    total_subpackets += len(subpackets)
                    valid_subpackets += sum(1 for s in subpackets if s["pkt_valid"])

                # Should be 100% valid
                self.assertEqual(
                    valid_subpackets,
                    total_subpackets,
                    f"{preset}: {valid_subpackets}/{total_subpackets} valid",
                )

    def test_sensor_types_p20(self):
        """Test that p20 preset contains expected sensor types."""
        with open(self.test_files["p20"], "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Collect all sensor types
        sensor_types = set()
        for message in lines:
            if not message.strip():
                continue
            subpackets = parse_message(message)
            for subpkt in subpackets:
                if subpkt["pkt_valid"]:
                    sensor_types.add(subpkt["pkt_type"])

        # p20 should have EEG4, ACCGYRO, Battery, REF
        expected_types = {"EEG4", "ACCGYRO", "Battery", "REF"}
        self.assertTrue(
            expected_types.issubset(sensor_types),
            f"Expected {expected_types}, got {sensor_types}",
        )

    def test_sensor_types_p1034(self):
        """Test that p1034 preset contains expected sensor types."""
        with open(self.test_files["p1034"], "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Collect all sensor types
        sensor_types = set()
        for message in lines:
            if not message.strip():
                continue
            subpackets = parse_message(message)
            for subpkt in subpackets:
                if subpkt["pkt_valid"]:
                    sensor_types.add(subpkt["pkt_type"])

        # p1034 should have EEG4, Optics8, ACCGYRO, Battery, REF
        expected_types = {"EEG4", "Optics8", "ACCGYRO", "Battery", "REF"}
        self.assertTrue(
            expected_types.issubset(sensor_types),
            f"Expected {expected_types}, got {sensor_types}",
        )

    def test_sensor_types_p1041(self):
        """Test that p1041 preset contains expected sensor types."""
        with open(self.test_files["p1041"], "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Collect all sensor types
        sensor_types = set()
        for message in lines:
            if not message.strip():
                continue
            subpackets = parse_message(message)
            for subpkt in subpackets:
                if subpkt["pkt_valid"]:
                    sensor_types.add(subpkt["pkt_type"])

        # p1041 should have EEG8, Optics16, ACCGYRO, Battery, REF
        expected_types = {"EEG8", "Optics16", "ACCGYRO", "Battery", "REF"}
        self.assertTrue(
            expected_types.issubset(sensor_types),
            f"Expected {expected_types}, got {sensor_types}",
        )

    def test_all_presets_comprehensive(self):
        """Comprehensive test: validate all 15 presets for coverage and validity.

        Tests all messages in each preset file to ensure complete coverage including
        infrequent sensor types (Battery ~60s intervals).
        """
        # Expected sensor configurations per preset (all messages)
        preset_expectations = {
            "p20": {"EEG4", "ACCGYRO", "REF", "Battery"},
            "p21": {"EEG4", "ACCGYRO", "REF", "Battery"},
            "p50": {"EEG4", "ACCGYRO", "REF", "Battery"},
            "p51": {"EEG4", "ACCGYRO", "REF", "Battery"},
            "p60": {"EEG4", "ACCGYRO", "REF", "Battery"},
            "p61": {"EEG4", "ACCGYRO", "REF", "Battery"},
            "p1034": {"EEG4", "Optics8", "ACCGYRO", "REF", "Battery"},
            "p1035": {"EEG4", "Optics4", "ACCGYRO", "REF", "Battery"},
            "p1041": {"EEG8", "Optics16", "ACCGYRO", "REF", "Battery"},
            "p1042": {"EEG8", "Optics16", "ACCGYRO", "REF", "Battery"},
            "p1043": {"EEG8", "Optics8", "ACCGYRO", "REF", "Battery"},
            "p1044": {"EEG8", "Optics8", "ACCGYRO", "REF", "Battery"},
            "p1045": {"EEG8", "Optics4", "ACCGYRO", "REF", "Battery"},
            "p1046": {"EEG8", "Optics4", "ACCGYRO", "REF", "Battery"},
            "p4129": {"EEG8", "Optics4", "ACCGYRO", "REF", "Battery"},
        }

        total_coverage_pass = 0
        total_validity_pass = 0

        for preset, filepath in self.test_files.items():
            with self.subTest(preset=preset):
                with open(filepath, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                total_payload_bytes = 0
                total_decoded_bytes = 0
                total_subpackets = 0
                valid_subpackets = 0
                sensor_types = set()

                for message in lines:  # Test ALL messages per preset
                    if not message.strip():
                        continue

                    # Extract payload size
                    parts = message.strip().split("\t", 2)
                    if len(parts) < 3:
                        continue
                    payload = bytes.fromhex(parts[2].strip())
                    payload_size = len(payload)

                    # Parse message
                    subpackets = parse_message(message)

                    # Calculate metrics
                    decoded_bytes = sum(s["pkt_len"] for s in subpackets)
                    total_payload_bytes += payload_size
                    total_decoded_bytes += decoded_bytes
                    total_subpackets += len(subpackets)
                    valid_subpackets += sum(1 for s in subpackets if s["pkt_valid"])

                    # Collect sensor types
                    for subpkt in subpackets:
                        if subpkt["pkt_valid"]:
                            sensor_types.add(subpkt["pkt_type"])

                # Test coverage
                coverage = 100.0 * total_decoded_bytes / total_payload_bytes
                self.assertEqual(
                    coverage,
                    100.0,
                    f"{preset}: Coverage {coverage:.2f}% (expected 100%)",
                )
                if coverage == 100.0:
                    total_coverage_pass += 1

                # Test validity
                validity = 100.0 * valid_subpackets / total_subpackets
                self.assertEqual(
                    validity,
                    100.0,
                    f"{preset}: Validity {validity:.2f}% (expected 100%)",
                )
                if validity == 100.0:
                    total_validity_pass += 1

                # Test sensor types
                expected = preset_expectations.get(preset, set())
                self.assertTrue(
                    expected.issubset(sensor_types),
                    f"{preset}: Expected {expected}, got {sensor_types}",
                )

        # Overall summary
        print(f"\n  ✓ All {len(self.test_files)} presets: 100% coverage")
        print(f"  ✓ All {len(self.test_files)} presets: 100% validity")

    def test_counter_increment(self):
        """Test that packet counters increment correctly (with wrap-around)."""
        with open(self.test_files["p20"], "r", encoding="utf-8") as f:
            lines = f.readlines()

        prev_counter = None
        for message in lines[:20]:  # Test first 20 messages
            if not message.strip():
                continue

            subpackets = parse_message(message)
            for subpkt in subpackets:
                if not subpkt["pkt_valid"]:
                    continue

                counter = subpkt["pkt_n"]
                if prev_counter is not None:
                    # Should increment by 1 (with wrap at 255->0)
                    expected = (prev_counter + 1) % 256
                    self.assertEqual(
                        counter,
                        expected,
                        f"Counter jump: {prev_counter} -> {counter}, expected {expected}",
                    )
                prev_counter = counter

    def test_timestamp_increasing(self):
        """Test that device timestamps are mostly increasing."""
        with open(self.test_files["p20"], "r", encoding="utf-8") as f:
            lines = f.readlines()

        timestamps = []
        for message in lines[:50]:  # Test first 50 messages
            if not message.strip():
                continue

            subpackets = parse_message(message)
            for subpkt in subpackets:
                if not subpkt["pkt_valid"]:
                    continue
                timestamps.append(subpkt["pkt_time"])

        # Check that most timestamps are increasing (allow some out-of-order)
        increasing_count = sum(
            1 for i in range(1, len(timestamps)) if timestamps[i] > timestamps[i - 1]
        )
        increasing_ratio = increasing_count / (len(timestamps) - 1)

        # At least 90% should be increasing (BLE packets can arrive slightly out-of-order)
        self.assertGreater(
            increasing_ratio,
            0.90,
            f"Only {increasing_ratio:.1%} of timestamps are increasing",
        )

    def test_performance(self):
        """Test that parsing is fast enough for real-time use."""
        import time

        with open(self.test_files["p20"], "r", encoding="utf-8") as f:
            lines = f.readlines()[:100]  # Test 100 messages

        start_time = time.perf_counter()
        total_subpackets = 0

        for message in lines:
            if not message.strip():
                continue
            subpackets = parse_message(message)
            total_subpackets += len(subpackets)

        elapsed = time.perf_counter() - start_time
        avg_time_per_message = elapsed / len(lines) * 1000  # Convert to ms

        # Should be faster than 1 ms per message for real-time suitability
        self.assertLess(
            avg_time_per_message,
            1.0,
            f"Parsing too slow: {avg_time_per_message:.3f} ms per message",
        )

        # Print performance info
        print(
            f"\nPerformance: {avg_time_per_message:.3f} ms/msg, "
            f"{total_subpackets} subpackets in {elapsed:.3f} sec"
        )

    def test_empty_payload(self):
        """Test handling of empty or malformed messages."""
        # Test with various malformed inputs
        malformed = [
            "2025-09-25T08:00:38.601520Z\t273e0013-4c4d-454d-96be-f03bac821358\t",  # Empty hex
            "2025-09-25T08:00:38.601520Z\t273e0013-4c4d-454d-96be-f03bac821358\t00",  # Too short
            "2025-09-25T08:00:38.601520Z\t273e0013-4c4d-454d-96be-f03bac821358\t0a0102030405060708090a0b0c",  # 13 bytes
            "malformed\tmessage",  # Not enough fields
            "invalid-timestamp\tuuid\thexstring",  # Invalid timestamp
        ]

        for msg in malformed:
            with self.subTest(msg=msg[:50]):
                # Should not raise, just return empty or incomplete list
                subpackets = parse_message(msg)
                # Should return empty list or only invalid packets
                if subpackets:
                    # If any packets returned, they should be invalid
                    valid_count = sum(
                        1 for s in subpackets if s.get("pkt_valid", False)
                    )
                    self.assertEqual(
                        valid_count,
                        0,
                        f"Got {valid_count} valid packets from malformed message",
                    )

    def test_data_section_length(self):
        """Test that data section length matches expected size."""
        with open(self.test_files["p20"], "r", encoding="utf-8") as f:
            message = f.readline()

        subpackets = parse_message(message)
        for subpkt in subpackets:
            if not subpkt["pkt_valid"]:
                continue

            # Data length should be pkt_len - 14 (header size)
            expected_data_len = subpkt["pkt_len"] - 14
            actual_data_len = len(subpkt["pkt_data"])

            self.assertEqual(
                actual_data_len,
                expected_data_len,
                f"Data length mismatch: expected {expected_data_len}, got {actual_data_len}",
            )


class TestLookupTables(unittest.TestCase):
    """Test the FREQ_MAP and TYPE_MAP lookup tables."""

    def test_freq_map_completeness(self):
        """Test that FREQ_MAP contains expected frequency codes."""
        expected_codes = {1, 2, 3, 4, 5, 6, 7, 8, 9}
        actual_codes = set(FREQ_MAP.keys())
        self.assertEqual(actual_codes, expected_codes)

    def test_type_map_completeness(self):
        """Test that TYPE_MAP contains expected type codes."""
        expected_codes = {1, 2, 3, 4, 5, 6, 7, 8}
        actual_codes = set(TYPE_MAP.keys())
        self.assertEqual(actual_codes, expected_codes)

    def test_freq_map_values(self):
        """Test that FREQ_MAP values are reasonable frequencies."""
        for code, freq in FREQ_MAP.items():
            self.assertIsInstance(freq, float)
            self.assertGreater(freq, 0)
            self.assertLessEqual(freq, 300)  # Reasonable upper bound

    def test_type_map_values(self):
        """Test that TYPE_MAP values are valid sensor types."""
        expected_types = {
            "EEG4",
            "EEG8",
            "REF",
            "Optics4",
            "Optics8",
            "Optics16",
            "ACCGYRO",
            "Battery",
        }
        actual_types = set(TYPE_MAP.values())
        self.assertEqual(actual_types, expected_types)


class TestDecodeBattery(unittest.TestCase):
    """Test suite for decode_battery() function integrated with parse_message()."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures - locate all test data files."""
        cls.test_data_dir = os.path.join(os.path.dirname(__file__), "test_data")

        # All available test data files
        cls.all_test_files = [
            "data_p20.txt",
            "data_p21.txt",
            "data_p50.txt",
            "data_p51.txt",
            "data_p60.txt",
            "data_p61.txt",
            "data_p1034.txt",
            "data_p1035.txt",
            "data_p1041.txt",
            "data_p1042.txt",
            "data_p1043.txt",
            "data_p1044.txt",
            "data_p1045.txt",
            "data_p1046.txt",
            "data_p4129.txt",
        ]

        # Battery test files with known battery levels
        cls.battery_test_files = {
            "16.80%": os.path.join(cls.test_data_dir, "test_battery_16_80.txt"),
            "90.40%": os.path.join(cls.test_data_dir, "test_battery_90_40.txt"),
            "58.27%": os.path.join(cls.test_data_dir, "test_battery_58_27.txt"),
        }

        # Verify battery test files exist
        for label, filepath in cls.battery_test_files.items():
            if not os.path.exists(filepath):
                raise FileNotFoundError(
                    f"Battery test file not found: {filepath}\n"
                    f"Please ensure the file exists in {cls.test_data_dir}"
                )

        # Keep backward compatibility
        cls.battery_file_16_80 = cls.battery_test_files["16.80%"]

    def test_battery_monotonic_decrease_all_files(self):
        """Test that battery values monotonically decrease across all test files.

        Validates:
        1. Battery never increases
        2. Change rate is reasonable (no more than ~10% per 30 seconds)
        """
        for filename in self.all_test_files:
            filepath = os.path.join(self.test_data_dir, filename)
            if not os.path.exists(filepath):
                continue

            with self.subTest(file=filename):
                with open(filepath, "r", encoding="utf-8") as f:
                    messages = f.readlines()

                # Collect battery readings with timestamps
                battery_data = []  # List of (time, battery_percent)
                for message in messages:
                    subpackets = parse_message(message)
                    for sp in subpackets:
                        if sp["pkt_type"] == "Battery":
                            times, data = sp["pkt_data"]
                            # times and data are single-element arrays
                            battery_data.append((times[0], data[0]))

                if len(battery_data) < 2:
                    # Skip files without sufficient battery data
                    continue

                # Sort by time to ensure chronological order
                battery_data.sort(key=lambda x: x[0])

                # Check monotonic decrease
                for i in range(1, len(battery_data)):
                    time_prev, battery_prev = battery_data[i - 1]
                    time_curr, battery_curr = battery_data[i]

                    # Battery should never increase
                    self.assertLessEqual(
                        battery_curr,
                        battery_prev,
                        f"{filename}: Battery increased at {time_curr:.3f}s: "
                        f"{battery_prev:.2f}% -> {battery_curr:.2f}%",
                    )

                    # Check change rate: no more than ~10% per 30 seconds
                    time_diff = time_curr - time_prev
                    battery_diff = battery_prev - battery_curr

                    if time_diff > 0:
                        # Expected max rate: 10% per 30s = 0.333% per second
                        max_expected_change = (time_diff / 30.0) * 10.0
                        self.assertLessEqual(
                            battery_diff,
                            max_expected_change,
                            f"{filename}: Battery dropped too fast over {time_diff:.1f}s: "
                            f"{battery_prev:.2f}% -> {battery_curr:.2f}% "
                            f"(change: {battery_diff:.2f}%, max expected: {max_expected_change:.2f}%)",
                        )

    def test_battery_known_levels(self):
        """Test battery decoding with files at known battery levels."""
        expected_levels = {
            "16.80%": 16.80,
            "90.40%": 90.40,
            "58.27%": 58.27,
        }

        for label, filepath in self.battery_test_files.items():
            with self.subTest(battery_level=label):
                with open(filepath, "r", encoding="utf-8") as f:
                    messages = f.readlines()

                # Parse messages and extract battery percentages
                battery_values = []
                for message in messages:
                    subpackets = parse_message(message)
                    for sp in subpackets:
                        if sp["pkt_type"] == "Battery":
                            times, data = sp["pkt_data"]
                            battery_values.append(
                                data[0]
                            )  # data is single-element array

                # Verify we found battery packets
                self.assertGreater(
                    len(battery_values), 0, f"{label}: Should find battery packets"
                )

                # Verify battery percentage average is close to expected
                avg_battery = sum(battery_values) / len(battery_values)
                expected = expected_levels[label]
                self.assertAlmostEqual(
                    avg_battery,
                    expected,
                    delta=0.5,  # Allow 0.5% tolerance
                    msg=f"{label}: Average battery should be ~{expected:.2f}%, got {avg_battery:.2f}%",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)

