"""
Unit tests for the new decode module.

Tests the parse_message() and decode_rawdata() functions with real BLE message data from Muse devices.
The new decoder returns numpy arrays with timestamps instead of lists of dicts.
"""

import unittest
import os
import numpy as np
from OpenMuse.decode import parse_message, decode_rawdata, SENSORS


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

        result = parse_message(message)

        # Should return a dict
        self.assertIsInstance(result, dict)

        # Should have all sensor type keys
        expected_keys = {"EEG", "ACCGYRO", "Optics", "Battery", "Unknown"}
        self.assertEqual(set(result.keys()), expected_keys)

        # Each value should be a numpy array
        for key, value in result.items():
            self.assertIsInstance(value, np.ndarray)

    def test_accgyro_structure(self):
        """Test that ACCGYRO data has correct structure."""
        with open(self.test_files["p20"], "r", encoding="utf-8") as f:
            message = f.readline()

        result = parse_message(message)
        accgyro = result["ACCGYRO"]

        # Should have data (p20 has ACCGYRO)
        self.assertGreater(accgyro.shape[0], 0)

        # Should have 7 columns: time + 6 channels (ACC_X, ACC_Y, ACC_Z, GYRO_X, GYRO_Y, GYRO_Z)
        self.assertEqual(accgyro.shape[1], 7)

        # First column should be timestamps (increasing)
        timestamps = accgyro[:, 0]
        self.assertTrue(
            np.all(timestamps[1:] >= timestamps[:-1]),
            "Timestamps should be monotonically increasing",
        )

    def test_eeg_structure(self):
        """Test that EEG data has correct structure."""
        with open(self.test_files["p1034"], "r", encoding="utf-8") as f:
            message = f.readline()

        result = parse_message(message)
        eeg = result["EEG"]

        # Should have data (p1034 has EEG)
        self.assertGreater(eeg.shape[0], 0)

        # Should have time + channels (either 4 or 8)
        self.assertIn(eeg.shape[1], [5, 9])  # time + 4 or time + 8

        # First column should be timestamps
        timestamps = eeg[:, 0]
        self.assertTrue(np.all(timestamps[1:] >= timestamps[:-1]))

    def test_empty_message(self):
        """Test handling of empty or malformed messages."""
        result = parse_message("")

        # Should return empty arrays for all sensor types
        for key, value in result.items():
            self.assertIsInstance(value, np.ndarray)
            self.assertEqual(value.shape, (0, 0))

    def test_timestamp_uniformity(self):
        """Test that timestamps are uniformly spaced at correct sampling rate."""
        with open(self.test_files["p20"], "r", encoding="utf-8") as f:
            message = f.readline()

        result = parse_message(message)
        accgyro = result["ACCGYRO"]

        if accgyro.shape[0] > 1:
            timestamps = accgyro[:, 0]
            deltas = np.diff(timestamps)

            # Expected delta for 52 Hz ACCGYRO
            expected_delta = 1.0 / 52.0

            # Check that deltas are very close to expected (within 1%)
            self.assertTrue(np.allclose(deltas, expected_delta, rtol=0.01))

    def test_sensors_config(self):
        """Test that SENSORS configuration is complete."""
        # Should have all expected sensor TAG bytes
        expected_tags = {0x11, 0x12, 0x34, 0x35, 0x36, 0x47, 0x53, 0x98}
        self.assertEqual(set(SENSORS.keys()), expected_tags)

        # Each sensor should have required fields
        for tag, config in SENSORS.items():
            self.assertIn("type", config)
            self.assertIn("n_channels", config)
            self.assertIn("n_samples", config)
            self.assertIn("rate", config)
            self.assertIn("data_len", config)

            # Validate types
            self.assertIsInstance(config["type"], str)
            self.assertIsInstance(config["n_channels"], int)
            self.assertIsInstance(config["n_samples"], int)
            self.assertIsInstance(config["rate"], float)
            self.assertIsInstance(config["data_len"], int)

    def test_performance(self):
        """Test that parsing is fast enough for real-time use."""
        import time

        # Read 100 messages
        messages = []
        with open(self.test_files["p20"], "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 100:
                    break
                messages.append(line)

        # Time the parsing
        start = time.perf_counter()
        for message in messages:
            parse_message(message)
        elapsed = time.perf_counter() - start

        # Should be fast: < 1ms per message on average
        ms_per_msg = (elapsed / len(messages)) * 1000
        self.assertLess(ms_per_msg, 1.0, f"Parsing too slow: {ms_per_msg:.3f} ms/msg")
        print(f"\nPerformance: {ms_per_msg:.3f} ms/msg")


class TestDecodeRawdata(unittest.TestCase):
    """Test suite for decode_rawdata() function."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.test_data_dir = os.path.join(os.path.dirname(__file__), "test_data")
        cls.test_file = os.path.join(cls.test_data_dir, "data_p20.txt")

    def test_basic_decoding(self):
        """Test basic multi-message decoding to DataFrames."""
        # Read 10 messages
        with open(self.test_file, "r", encoding="utf-8") as f:
            messages = [f.readline() for _ in range(10)]

        result = decode_rawdata(messages)

        # Should return a dict
        self.assertIsInstance(result, dict)

        # Should have all sensor type keys
        expected_keys = {"EEG", "ACCGYRO", "Optics", "Battery", "Unknown"}
        self.assertEqual(set(result.keys()), expected_keys)

        # Each value should be a DataFrame
        import pandas as pd

        for key, value in result.items():
            self.assertIsInstance(value, pd.DataFrame)

    def test_accgyro_columns(self):
        """Test that ACCGYRO DataFrame has correct columns."""
        with open(self.test_file, "r", encoding="utf-8") as f:
            messages = [f.readline() for _ in range(10)]

        result = decode_rawdata(messages)
        accgyro_df = result["ACCGYRO"]

        # Should have data
        self.assertGreater(len(accgyro_df), 0)

        # Should have correct columns
        expected_columns = [
            "time",
            "ACC_X",
            "ACC_Y",
            "ACC_Z",
            "GYRO_X",
            "GYRO_Y",
            "GYRO_Z",
        ]
        self.assertEqual(list(accgyro_df.columns), expected_columns)

    def test_eeg_columns(self):
        """Test that EEG DataFrame has correct columns."""
        # Use p1034 which has EEG4 data
        test_file = os.path.join(self.test_data_dir, "data_p1034.txt")
        with open(test_file, "r", encoding="utf-8") as f:
            messages = [f.readline() for _ in range(10)]

        result = decode_rawdata(messages)
        eeg_df = result["EEG"]

        # Should have data
        if len(eeg_df) > 0:
            # Should have correct columns for EEG4
            expected_columns = ["time", "EEG_TP9", "EEG_AF7", "EEG_AF8", "EEG_TP10"]
            self.assertEqual(list(eeg_df.columns), expected_columns)

    def test_optics_columns(self):
        """Test that Optics DataFrame has correct columns."""
        # Use p1041 which has Optics16 data
        test_file = os.path.join(self.test_data_dir, "data_p1041.txt")
        with open(test_file, "r", encoding="utf-8") as f:
            messages = [f.readline() for _ in range(10)]

        result = decode_rawdata(messages)
        optics_df = result["Optics"]

        # Should have data
        if len(optics_df) > 0:
            # Should have correct columns for Optics16
            expected_columns = [
                "time",
                "OPTICS_LO_NIR",
                "OPTICS_RO_NIR",
                "OPTICS_LO_IR",
                "OPTICS_RO_IR",
                "OPTICS_LI_NIR",
                "OPTICS_RI_NIR",
                "OPTICS_LI_IR",
                "OPTICS_RI_IR",
                "OPTICS_LO_RED",
                "OPTICS_RO_RED",
                "OPTICS_LO_AMB",
                "OPTICS_RO_AMB",
                "OPTICS_LI_RED",
                "OPTICS_RI_RED",
                "OPTICS_LI_AMB",
                "OPTICS_RI_AMB",
            ]
            self.assertEqual(list(optics_df.columns), expected_columns)

    def test_concatenation(self):
        """Test that multiple messages are properly concatenated."""
        # Parse 5 messages individually
        individual_arrays = []
        with open(self.test_file, "r", encoding="utf-8") as f:
            for _ in range(5):
                message = f.readline()
                result = parse_message(message)
                if result["ACCGYRO"].shape[0] > 0:
                    individual_arrays.append(result["ACCGYRO"])

        # Total samples from individual parsing
        individual_total = sum(arr.shape[0] for arr in individual_arrays)

        # Parse all messages together
        with open(self.test_file, "r", encoding="utf-8") as f:
            messages = [f.readline() for _ in range(5)]

        result = decode_rawdata(messages)
        combined_total = len(result["ACCGYRO"])

        # Should have same total samples
        self.assertEqual(individual_total, combined_total)


class TestBattery(unittest.TestCase):
    """Test suite for battery decoding."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.test_data_dir = os.path.join(os.path.dirname(__file__), "test_data")
        cls.battery_files = {
            "16_80": os.path.join(cls.test_data_dir, "test_battery_16_80.txt"),
            "58_27": os.path.join(cls.test_data_dir, "test_battery_58_27.txt"),
            "90_40": os.path.join(cls.test_data_dir, "test_battery_90_40.txt"),
        }

    def test_battery_known_levels(self):
        """Test battery decoding with files at known battery levels."""
        expected_values = {
            "16_80": 16.80,
            "58_27": 58.27,
            "90_40": 90.40,
        }

        for key, filepath in self.battery_files.items():
            # Read all messages and collect battery data
            battery_values = []
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    result = parse_message(line)
                    battery = result["Battery"]
                    if battery.shape[0] > 0:
                        battery_values.append(battery[0, 1])

            # Should have found at least one battery sample
            self.assertGreater(
                len(battery_values), 0, f"No battery data found in {key}"
            )

            # Check first battery value
            battery_percent = battery_values[0]
            expected = expected_values[key]

            # Should be within 0.5% of expected
            self.assertAlmostEqual(
                battery_percent,
                expected,
                delta=0.5,
                msg=f"Battery level mismatch for {key}: got {battery_percent:.2f}, expected {expected:.2f}",
            )


if __name__ == "__main__":
    unittest.main()
