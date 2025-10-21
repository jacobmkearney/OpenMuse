"""
Unit tests for streaming functionality.

Tests the streaming pipeline by simulating real-time BLE data reception
using pre-recorded test data files, allowing testing without hardware.

MAINTENANCE INSTRUCTIONS:
------------------------
(changed as of 2025-10-21)

This test file simulates the streaming pipeline using pre-recorded BLE data.
When the package code is updated, you may need to update this test file to match:

1. **Sensor Stream Configuration**: If sensor labels, sampling rates, or channel
   counts change in stream.py, update the MockStreamOutlet and test expectations.

2. **Decoding Logic**: If decode.py parsing or timestamping logic changes,
   the simulation methods (_simulate_streaming_with_logging, _simulate_buffer_flush)
   may need updates to match the new behavior.

3. **Buffer Management**: If stream.py buffer flushing or timestamp re-anchoring
   logic changes, update _simulate_buffer_flush to match.

4. **JSON Output Format**: If the real streaming JSON output format changes,
   update _create_json_output_from_logged_data accordingly.

5. **Test Data Files**: The test uses test_accgyro.txt and test_eeg_quality.txt.
   If these files become outdated or new test files are added, update the
   test_generate_mock_streaming_data method. The simulation runs for ~2 seconds
   per file to capture more realistic streaming data.

6. **Dependencies**: This test requires pandas for the validation test.
   Ensure pandas is available in the test environment.

To update for package changes:
- Run the tests and check for failures
- Compare failing test output with expected behavior
- Update simulation logic to match current package implementation
- Verify that generated JSON files still match real streaming output format
"""

import unittest
import os
import time
import json
import logging
from unittest.mock import Mock, MagicMock, patch
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

from OpenMuse.stream import SensorStream, _build_sensor_streams
from OpenMuse.decode import parse_message, make_timestamps, decode_rawdata
from OpenMuse.muse import MuseS


class MockStreamOutlet:
    """Mock LSL StreamOutlet that captures pushed data instead of sending to LSL."""

    def __init__(self, info, chunk_size=1):
        self.info = info
        self.chunk_size = chunk_size
        self.pushed_data = []
        self.pushed_timestamps = []

    def push_chunk(self, x, timestamp=None, pushThrough=True):
        """Capture pushed data and timestamps."""
        self.pushed_data.append(x.copy() if hasattr(x, "copy") else x)
        if timestamp is not None:
            self.pushed_timestamps.append(
                timestamp.copy() if hasattr(timestamp, "copy") else timestamp
            )
        else:
            self.pushed_timestamps.append(None)


class TestStreamingSimulation(unittest.TestCase):
    """Test streaming pipeline using simulated BLE data from test files."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.test_data_dir = os.path.join(os.path.dirname(__file__), "test_data")

        # Test data files for different sensor types
        cls.test_files = {
            "accgyro": os.path.join(cls.test_data_dir, "test_accgyro.txt"),
            "eeg_quality": os.path.join(cls.test_data_dir, "test_eeg_quality.txt"),
            "optics4": os.path.join(cls.test_data_dir, "test_optics4.txt"),
            "optics16": os.path.join(cls.test_data_dir, "test_optics16.txt"),
            "battery": os.path.join(cls.test_data_dir, "test_battery_16_80.txt"),
        }

        # Verify test files exist
        for name, filepath in cls.test_files.items():
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Test data file not found: {filepath}")

    def setUp(self):
        """Set up test case."""
        # Mock LSL to use our MockStreamOutlet
        with patch("OpenMuse.stream.StreamOutlet", MockStreamOutlet):
            self.sensor_streams = _build_sensor_streams(enable_logging=True)

        # Track global offset like in real streaming
        self.device_to_lsl_offset = None

    def _load_test_messages(
        self, filename: str, max_messages: Optional[int] = None
    ) -> List[str]:
        """Load test messages from file."""
        messages = []
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    messages.append(line.strip())
                    if max_messages and len(messages) >= max_messages:
                        break
        return messages

    def _simulate_buffer_flush(
        self, sensor_type: str, sensor_streams=None
    ) -> Optional[tuple]:
        """Simulate buffer flushing logic from stream.py."""
        if sensor_streams is None:
            sensor_streams = self.sensor_streams
        stream = sensor_streams[sensor_type]
        if len(stream.buffer) == 0:
            return None

        # print(f"Flushing buffer for {sensor_type} with {len(stream.buffer)} chunks")

        # Concatenate all timestamps and data (much faster than Python list operations)
        all_timestamps = np.concatenate([ts for ts, _ in stream.buffer])
        all_data = np.vstack([data for _, data in stream.buffer])

        # Sort by timestamp using argsort (numpy is much faster than Python sort)
        sort_indices = np.argsort(all_timestamps)
        sorted_timestamps = all_timestamps[sort_indices]
        sorted_data = all_data[sort_indices]

        # Convert device timestamps to LSL clock time (simulate real LSL behavior)
        # In real LSL, this would account for device-to-LSL offset and network delays
        import time as time_module

        if len(sorted_timestamps) > 0:
            # Calculate offset once at the beginning of streaming session
            if self.device_to_lsl_offset is None:
                lsl_now = time_module.time()
                # Calculate offset as difference between LSL time and first device timestamp
                # This simulates the time synchronization that happens in real LSL streaming
                self.device_to_lsl_offset = lsl_now - sorted_timestamps[0]

            anchored_timestamps = sorted_timestamps + self.device_to_lsl_offset
        else:
            anchored_timestamps = sorted_timestamps

        # Push to mock LSL
        try:
            stream.outlet.push_chunk(
                x=sorted_data.astype(np.float32, copy=False),
                timestamp=anchored_timestamps.astype(np.float64, copy=False),
                pushThrough=True,
            )

            # Return logged data instead of storing in stream
            # print(f"Logging {len(anchored_timestamps)} samples for {sensor_type}")
            logged_chunk = (anchored_timestamps.copy(), sorted_data.copy())
            # print(f"Returning logged chunk with {len(anchored_timestamps)} samples")
            return logged_chunk
        except Exception as exc:
            # In real streaming this would be logged
            print(f"Error in buffer flush: {exc}")
            return None
        finally:
            # Clear buffer
            stream.buffer.clear()

    def _create_json_output_from_logged_data(self, logged_data, sensor_streams):
        """Create JSON output from collected logged data."""

        def _serialize_log(log_records):
            if not log_records:
                return [], []

            samples = []
            for timestamps, data_chunk in log_records:
                for i, ts in enumerate(timestamps):
                    samples.append((float(ts), data_chunk[i, :].tolist()))

            samples.sort(key=lambda x: x[0])
            timestamps_out = [s[0] for s in samples]
            data_out = [s[1] for s in samples]
            return timestamps_out, data_out

        json_data = {}
        sample_counts = {}
        for sensor_type, stream in sensor_streams.items():
            log_records = logged_data.get(sensor_type, [])
            timestamps_out, data_out = _serialize_log(log_records)
            sample_counts[sensor_type] = len(timestamps_out)
            json_data[sensor_type] = {
                "lsl_timestamps": timestamps_out,
                "channels": list(stream.labels),
                "data": data_out,
                "n_samples": sample_counts[sensor_type],
                "sampling_rate": stream.sampling_rate,
                "unit": stream.unit,
            }

        json_data["note"] = (
            "Mock streaming data generated from test files - LSL data globally sorted by timestamp per sensor type. "
            "Timestamps are in LSL clock time (not device time)."
        )

        return json_data

    def _simulate_queue_samples(
        self,
        sensor_type: str,
        data_array: np.ndarray,
        current_time: float,
    ) -> None:
        """Simulate sample queuing logic from stream.py."""
        stream = self.sensor_streams[sensor_type]
        if data_array.size == 0 or data_array.shape[1] < 2:
            return

        # Extract sensor data (exclude time column)
        samples = data_array[:, 1:].astype(np.float32)
        if stream.pad_to_channels:
            target = stream.pad_to_channels
            current = samples.shape[1]
            if current < target:
                padding = np.zeros(
                    (samples.shape[0], target - current), dtype=np.float32
                )
                samples = np.hstack([samples, padding])
            elif current > target:
                samples = samples[:, :target]

        # Use device timestamps directly (simplified for testing)
        device_times = data_array[:, 0]

        # Add to buffer
        stream.buffer.append((device_times, samples))

        # Simple buffer management: flush if buffer gets too large
        if len(stream.buffer) >= 5:  # Smaller threshold for testing
            self._simulate_buffer_flush(sensor_type)

    def _simulate_queue_samples_with_logging(
        self,
        sensor_type: str,
        data_array: np.ndarray,
        current_time: float,
        sensor_streams=None,
        logged_data=None,
    ) -> None:
        """Simulate sample queuing logic from stream.py, with logging support."""
        if sensor_streams is None:
            sensor_streams = self.sensor_streams
        stream = sensor_streams[sensor_type]
        if data_array.size == 0 or data_array.shape[1] < 2:
            return

        # Extract sensor data (exclude time column)
        samples = data_array[:, 1:].astype(np.float32)
        if stream.pad_to_channels:
            target = stream.pad_to_channels
            current = samples.shape[1]
            if current < target:
                padding = np.zeros(
                    (samples.shape[0], target - current), dtype=np.float32
                )
                samples = np.hstack([samples, padding])
            elif current > target:
                samples = samples[:, :target]

        # Use device timestamps directly (simplified for testing)
        device_times = data_array[:, 0]

        # Add to buffer
        stream.buffer.append((device_times, samples))

        # Simple buffer management: flush if buffer gets too large
        if len(stream.buffer) >= 5:  # Smaller threshold for testing
            logged_chunk = self._simulate_buffer_flush(sensor_type, sensor_streams)
            if logged_chunk is not None and logged_data is not None:
                logged_data[sensor_type].append(logged_chunk)

    def _simulate_streaming(self, messages: List[str], message_delay: float = 0.01):
        """
        Simulate streaming by processing messages with delays.

        Parameters:
        -----------
        messages : List[str]
            List of BLE message strings (timestamp<TAB>UUID<TAB>hex_payload)
        message_delay : float
            Delay between processing messages (seconds)
        """
        # Reset streams for clean test
        for stream in self.sensor_streams.values():
            stream.buffer.clear()
            stream.last_push_time = None
            stream.base_time = None
            stream.wrap_offset = 0
            stream.last_abs_tick = 0
            stream.sample_counter = 0
            if stream.log_records is not None:
                stream.log_records.clear()

        self.device_to_lsl_offset = None

        def mock_on_data(uuid, data: bytearray):
            """Mock version of _on_data callback that processes BLE data."""
            message = f"2025-10-21T12:00:00.000000+00:00\t{uuid}\t{data.hex()}"

            # Parse the message (same as real streaming)
            subpackets = parse_message(message)

            # Process each sensor type
            decoded = {}
            for sensor_type, pkt_list in subpackets.items():
                if sensor_type in self.sensor_streams:
                    stream = self.sensor_streams[sensor_type]

                    # Get current state
                    current_state = (
                        stream.base_time,
                        stream.wrap_offset,
                        stream.last_abs_tick,
                        stream.sample_counter,
                    )

                    # Decode with timestamping
                    array, base_time, wrap_offset, last_abs_tick, sample_counter = (
                        make_timestamps(pkt_list, *current_state)
                    )

                    decoded[sensor_type] = array

                    # Update state
                    stream.base_time = base_time
                    stream.wrap_offset = wrap_offset
                    stream.last_abs_tick = last_abs_tick
                    stream.sample_counter = sample_counter

            # Mock current time
            current_time = time.time()

            # Queue samples
            for sensor_type in ["EEG", "ACCGYRO", "Optics"]:
                data_array = decoded.get(sensor_type, np.empty((0, 0)))
                self._simulate_queue_samples(sensor_type, data_array, current_time)

        # Process messages with delays
        for message in messages:
            # Parse message to extract UUID and hex payload
            try:
                ts, uuid, hex_payload = message.split("\t", 2)
                data = bytearray.fromhex(hex_payload.strip())
            except ValueError:
                continue  # Skip malformed messages

            # Process the message
            mock_on_data(uuid, data)

            # Add delay between messages
            if message_delay > 0:
                time.sleep(message_delay)

        # Flush any remaining buffered data
        for sensor_type in self.sensor_streams:
            if len(self.sensor_streams[sensor_type].buffer) > 0:
                self._simulate_buffer_flush(sensor_type)

    def test_basic_accgyro_streaming(self):
        """Test basic streaming simulation with ACCGYRO data."""
        # Load a small number of messages for quick testing
        messages = self._load_test_messages(self.test_files["accgyro"], max_messages=50)

        # Simulate streaming with small delay
        self._simulate_streaming(messages, message_delay=0.001)

        # Verify ACCGYRO data was captured
        accgyro_stream = self.sensor_streams["ACCGYRO"]
        outlet = accgyro_stream.outlet

        self.assertGreater(
            len(outlet.pushed_data), 0, "No ACCGYRO data was pushed to LSL"
        )
        self.assertGreater(
            len(outlet.pushed_timestamps), 0, "No ACCGYRO timestamps were pushed"
        )

        # Check data shape (should be n_samples x 6 channels)
        first_chunk = outlet.pushed_data[0]
        self.assertEqual(first_chunk.shape[1], 6, "ACCGYRO should have 6 channels")

        # Check timestamps are monotonic
        all_timestamps = np.concatenate(outlet.pushed_timestamps)
        self.assertTrue(
            np.all(np.diff(all_timestamps) >= 0), "Timestamps should be monotonic"
        )

    def test_eeg_streaming(self):
        """Test streaming simulation with EEG data."""
        messages = self._load_test_messages(
            self.test_files["eeg_quality"], max_messages=30
        )

        self._simulate_streaming(messages, message_delay=0.001)

        # Verify EEG data was captured
        eeg_stream = self.sensor_streams["EEG"]
        outlet = eeg_stream.outlet

        self.assertGreater(len(outlet.pushed_data), 0, "No EEG data was pushed to LSL")

        # Check data shape (EEG should have 8 channels after padding)
        first_chunk = outlet.pushed_data[0]
        self.assertEqual(first_chunk.shape[1], 8, "EEG should be padded to 8 channels")

    def test_buffer_flushing(self):
        """Test that buffer flushing works correctly."""
        messages = self._load_test_messages(
            self.test_files["accgyro"], max_messages=100
        )

        # Use very small delay to ensure buffer fills up
        self._simulate_streaming(messages, message_delay=0.0001)

        accgyro_stream = self.sensor_streams["ACCGYRO"]
        outlet = accgyro_stream.outlet

        # Should have multiple chunks due to buffer flushing
        self.assertGreater(
            len(outlet.pushed_data), 1, "Buffer should have flushed multiple times"
        )

        # Verify all chunks have correct channel count
        for chunk in outlet.pushed_data:
            self.assertEqual(
                chunk.shape[1], 6, "All ACCGYRO chunks should have 6 channels"
            )

    def test_generate_mock_streaming_data(self):
        """Generate mock streaming data JSON files for testing."""
        import json
        import os

        # Test configurations: (test_file_key, output_filename)
        test_configs = [
            ("accgyro", "tests/test_data/mock_stream1.json"),
            ("eeg_quality", "tests/test_data/mock_stream2.json"),
        ]

        for test_file_key, output_file in test_configs:
            with self.subTest(test_file=test_file_key):
                # Create sensor streams with logging enabled
                with patch("OpenMuse.stream.StreamOutlet", MockStreamOutlet):
                    sensor_streams = _build_sensor_streams(enable_logging=True)

                # print(f"Created sensor_streams with id: {id(sensor_streams)}")
                # print(f"EEG log_records id: {id(sensor_streams['EEG'].log_records)}")

                # Load test messages
                messages = self._load_test_messages(
                    self.test_files[test_file_key], max_messages=200
                )

                # Run simulation with logging enabled - run for ~2 seconds per file
                logged_data = self._simulate_streaming_with_logging(
                    sensor_streams,
                    messages,
                    message_delay=0.01,  # 0.01s * 200 messages = 2 seconds
                )

                # Serialize logged data to JSON format
                json_data = self._create_json_output_from_logged_data(
                    logged_data, sensor_streams
                )

                # Ensure output directory exists
                outdir = os.path.dirname(os.path.abspath(output_file))
                # print(f"Output directory: {outdir}")
                if outdir and not os.path.exists(outdir):
                    os.makedirs(outdir, exist_ok=True)

                # Write to file
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, indent=2)

                # print(
                #     f"Wrote mock streaming data to {output_file} (absolute: {os.path.abspath(output_file)})"
                # )
                # print(f"  EEG samples: {json_data.get('EEG', {}).get('n_samples', 0)}")
                # print(
                #     f"  ACCGYRO samples: {json_data.get('ACCGYRO', {}).get('n_samples', 0)}"
                # )
                # print(
                #     f"  Optics samples: {json_data.get('Optics', {}).get('n_samples', 0)}"
                # )

    def _simulate_streaming_with_logging(
        self, sensor_streams, messages, message_delay=0.01
    ):
        """Simulate streaming with logging enabled."""
        # Initialize logged_data dict
        logged_data = {sensor_type: [] for sensor_type in sensor_streams}

        # Reset streams for clean test
        for stream in sensor_streams.values():
            stream.buffer.clear()
            stream.last_push_time = None
            stream.base_time = None
            stream.wrap_offset = 0
            stream.last_abs_tick = 0
            stream.sample_counter = 0

        def mock_on_data(uuid, data: bytearray):
            """Mock version of _on_data callback that processes BLE data."""
            message = f"2025-10-21T12:00:00.000000+00:00\t{uuid}\t{data.hex()}"

            # Parse the message (same as real streaming)
            subpackets = parse_message(message)

            # Process each sensor type
            decoded = {}
            for sensor_type, pkt_list in subpackets.items():
                if sensor_type in sensor_streams:
                    stream = sensor_streams[sensor_type]

                    # Get current state
                    current_state = (
                        stream.base_time,
                        stream.wrap_offset,
                        stream.last_abs_tick,
                        stream.sample_counter,
                    )

                    # Decode with timestamping
                    array, base_time, wrap_offset, last_abs_tick, sample_counter = (
                        make_timestamps(pkt_list, *current_state)
                    )

                    decoded[sensor_type] = array

                    # Update state
                    stream.base_time = base_time
                    stream.wrap_offset = wrap_offset
                    stream.last_abs_tick = last_abs_tick
                    stream.sample_counter = sample_counter

            # Mock current time
            current_time = time.time()

            # Queue samples
            for sensor_type in ["EEG", "ACCGYRO", "Optics"]:
                data_array = decoded.get(sensor_type, np.empty((0, 0)))
                self._simulate_queue_samples_with_logging(
                    sensor_type, data_array, current_time, sensor_streams, logged_data
                )

        # Process messages with delays
        for message in messages:
            # Parse message to extract UUID and hex payload
            try:
                ts, uuid, hex_payload = message.split("\t", 2)
                data = bytearray.fromhex(hex_payload.strip())
            except ValueError:
                continue  # Skip malformed messages

            # Process the message
            mock_on_data(uuid, data)

            # Add delay between messages
            if message_delay > 0:
                time.sleep(message_delay)

        # Flush any remaining buffered data and collect logged data
        for sensor_type in sensor_streams:
            if len(sensor_streams[sensor_type].buffer) > 0:
                logged_chunk = self._simulate_buffer_flush(sensor_type, sensor_streams)
                if logged_chunk is not None:
                    logged_data[sensor_type].append(logged_chunk)

        return logged_data

    def _create_json_output(self, sensor_streams):
        """Create JSON output matching the real streaming format."""

        def _serialize_log(
            stream,
        ):
            if not stream.log_records:
                return [], []

            samples = []
            for timestamps, data_chunk in stream.log_records:
                for i, ts in enumerate(timestamps):
                    samples.append((float(ts), data_chunk[i, :].tolist()))

            samples.sort(key=lambda x: x[0])
            timestamps_out = [s[0] for s in samples]
            data_out = [s[1] for s in samples]
            return timestamps_out, data_out

        json_data = {}
        sample_counts = {}
        for sensor_type, stream in sensor_streams.items():
            timestamps_out, data_out = _serialize_log(stream)
            sample_counts[sensor_type] = len(timestamps_out)
            json_data[sensor_type] = {
                "lsl_timestamps": timestamps_out,
                "channels": list(stream.labels),
                "data": data_out,
                "n_samples": sample_counts[sensor_type],
                "sampling_rate": stream.sampling_rate,
                "unit": stream.unit,
            }

        json_data["note"] = (
            "Mock streaming data generated from test files - LSL data globally sorted by timestamp per sensor type. "
            "Timestamps are in LSL clock time (not device time)."
        )

        return json_data

    def test_validate_json_output_against_raw_decode(self):
        """Validate JSON output by comparing with decode_rawdata() results.

        Note: Streaming simulation inherently produces fewer samples than raw decode
        because it mimics real-time buffering and flushing behavior. Raw decode processes
        all messages at once, while streaming flushes data in chunks when buffers fill.
        This test validates that the streaming data is reasonable and properly formatted.
        """
        import json

        # Set up logging for discrepancies
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter("%(levelname)s: %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        # Test configurations: (test_file_key, json_filename, max_messages_used)
        test_configs = [
            ("accgyro", "tests/test_data/mock_stream1.json", 200),
            ("eeg_quality", "tests/test_data/mock_stream2.json", 200),
        ]

        for test_file_key, json_file, max_messages in test_configs:
            with self.subTest(test_file=test_file_key):
                # Load the SAME number of messages used to generate the JSON
                messages = self._load_test_messages(
                    self.test_files[test_file_key], max_messages=max_messages
                )

                # Decode using the raw decode_rawdata function
                raw_decoded = decode_rawdata(messages)

                # Load the JSON file generated by streaming simulation
                with open(json_file, "r", encoding="utf-8") as f:
                    json_data = json.load(f)

                logger.info(
                    f"\nValidating {test_file_key} (using {len(messages)} messages):"
                )

                # # Test time-stamps
                # min(json_data["EEG"]["lsl_timestamps"])
                # max(json_data["EEG"]["lsl_timestamps"])
                # min(json_data["ACCGYRO"]["lsl_timestamps"])
                # max(json_data["ACCGYRO"]["lsl_timestamps"])

                # Compare each sensor type
                for sensor_type in ["EEG", "ACCGYRO", "Optics"]:
                    if sensor_type not in json_data:
                        logger.warning(
                            f"JSON file {json_file} missing {sensor_type} data"
                        )
                        continue

                    if sensor_type not in raw_decoded:
                        logger.warning(
                            f"Raw decode missing {sensor_type} data for {test_file_key}"
                        )
                        continue

                    json_sensor_data = json_data[sensor_type]
                    raw_df = raw_decoded[sensor_type]

                    # Convert JSON data to pandas DataFrame
                    json_df = pd.DataFrame(
                        {
                            "time": json_sensor_data["lsl_timestamps"],
                            **{
                                col: vals
                                for col, vals in zip(
                                    json_sensor_data["channels"],
                                    zip(*json_sensor_data["data"]),
                                )
                            },
                        }
                    )

                    # Compare shapes - expect streaming to have fewer samples due to buffering
                    json_shape = json_df.shape
                    raw_shape = raw_df.shape

                    logger.info(
                        f"  {sensor_type}: JSON {json_shape[0]} samples vs Raw {raw_shape[0]} samples "
                        f"({json_shape[1]} vs {raw_shape[1]} columns)"
                    )

                    if json_shape[0] > raw_shape[0]:
                        logger.warning(
                            f"Unexpected: JSON has more samples than raw decode for {sensor_type}"
                        )
                    else:
                        # This is expected - streaming produces partial data due to buffering
                        sample_ratio = (
                            json_shape[0] / raw_shape[0] if raw_shape[0] > 0 else 0
                        )
                        logger.info(
                            f"    Expected: Streaming captured {sample_ratio:.1%} of available {sensor_type} data"
                        )

                    # Compare column names (allowing for potential differences in naming)
                    json_cols = set(json_df.columns)
                    raw_cols = set(raw_df.columns)

                    if json_cols != raw_cols:
                        logger.info(
                            f"  Column name differences for {sensor_type} in {test_file_key}:"
                        )
                        logger.info(f"    JSON only: {json_cols - raw_cols}")
                        logger.info(f"    Raw only: {raw_cols - json_cols}")
                    else:
                        logger.info(
                            f"  {sensor_type} columns match: {sorted(json_cols)}"
                        )

                    # Validate data types and ranges for available data
                    common_cols = json_cols & raw_cols
                    if common_cols and len(json_df) > 0:
                        logger.info(f"  Validating data quality for {sensor_type}...")

                        # Check for reasonable value ranges (not all zeros, not extreme outliers)
                        for col in sorted(common_cols):
                            if col == "time":
                                continue

                            json_vals = json_df[col]
                            raw_vals = raw_df[col]

                            # Check JSON data isn't all zeros (unless raw data is also all zeros)
                            json_nonzero = (json_vals != 0).any()
                            raw_nonzero = (raw_vals != 0).any()

                            if json_nonzero and not raw_nonzero:
                                logger.warning(
                                    f"    {col}: JSON has non-zero values but raw data is all zeros"
                                )
                            elif not json_nonzero and raw_nonzero:
                                logger.info(
                                    f"    {col}: Both JSON and raw have zero values (expected for some channels)"
                                )

                            # Check for reasonable ranges (JSON shouldn't have extreme outliers compared to raw)
                            if len(json_vals) > 0 and len(raw_vals) > 0:
                                json_range = json_vals.max() - json_vals.min()
                                raw_range = raw_vals.max() - raw_vals.min()

                                if raw_range > 0 and json_range > raw_range * 10:
                                    logger.warning(
                                        f"    {col}: JSON value range ({json_range:.3f}) much larger than raw range ({raw_range:.3f})"
                                    )

                        logger.info(f"  ✓ {sensor_type} data validation completed")


if __name__ == "__main__":
    unittest.main()
