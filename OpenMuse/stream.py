import asyncio
import json
import os
import time
from datetime import datetime, timezone
from typing import Optional

import bleak
import numpy as np

from .backends import _run
from .decode import parse_message
from .muse import MuseS

from mne_lsl.lsl import StreamInfo, StreamOutlet


def _build_outlets():
    """Build LSL outlets for EEG and ACC/GYRO data."""

    # EEG outlet - 4 channels at 256 Hz
    eeg_labels = ("TP9", "AF7", "AF8", "TP10")
    eeg_info = StreamInfo(
        name="Muse_EEG",
        stype="EEG",
        n_channels=4,
        sfreq=256.0,
        dtype="float32",
        source_id="Muse_EEG",
    )
    eeg_desc = eeg_info.desc
    eeg_desc.append_child_value("manufacturer", "Muse")
    eeg_channels = eeg_desc.append_child("channels")
    for label in eeg_labels:
        channel = eeg_channels.append_child("channel")
        channel.append_child_value("label", label)
        channel.append_child_value("unit", "microvolts")
        channel.append_child_value("type", "EEG")

    eeg_outlet = StreamOutlet(eeg_info, chunk_size=1)

    # ACC+GYRO outlet - 6 channels at 52 Hz
    accgyro_labels = ("ACC_X", "ACC_Y", "ACC_Z", "GYRO_X", "GYRO_Y", "GYRO_Z")
    accgyro_info = StreamInfo(
        name="Muse_ACCGYRO",
        stype="Motion",
        n_channels=6,
        sfreq=52.0,
        dtype="float32",
        source_id="Muse_ACCGYRO",
    )
    accgyro_desc = accgyro_info.desc
    accgyro_desc.append_child_value("manufacturer", "Muse")
    accgyro_channels = accgyro_desc.append_child("channels")
    for label in accgyro_labels:
        channel = accgyro_channels.append_child("channel")
        channel.append_child_value("label", label)
        channel.append_child_value("unit", "a.u.")

    accgyro_outlet = StreamOutlet(accgyro_info, chunk_size=1)

    return eeg_outlet, accgyro_outlet


async def _stream_async(
    address: str,
    preset: str,
    duration: Optional[float],
    outfile: Optional[str],
    verbose: bool,
) -> None:
    stream_started = asyncio.Event()
    eeg_outlet, accgyro_outlet = _build_outlets()
    samples_sent = {"EEG": 0, "ACCGYRO": 0}
    samples_written = 0

    # Compute device-to-LSL time offset once at the start
    # Will be updated with the first data packet
    device_to_lsl_offset = None

    # Reordering buffers for both EEG and ACC/GYRO data
    # Stores (lsl_timestamp, chunk_data) tuples
    eeg_buffer = []
    accgyro_buffer = []
    buffer_duration = 0.12  # Hold samples for 120ms to allow reordering (BLE can be delayed up to ~40ms)
    last_push_time = {"EEG": 0.0, "ACCGYRO": 0.0}

    def _ts() -> str:
        return datetime.now(timezone.utc).isoformat()

    # Collect LSL data for JSON output (exactly what gets pushed to LSL)
    # This will be populated during _flush_buffer() calls
    lsl_data_log = {"EEG": [], "ACCGYRO": []} if outfile else None

    def _flush_buffer(sensor_type: str):
        """Flush reordering buffer for a specific sensor type: sort and push samples to LSL."""
        nonlocal samples_sent, eeg_buffer, accgyro_buffer, last_push_time

        # Select appropriate buffer and outlet
        if sensor_type == "EEG":
            buffer = eeg_buffer
            outlet = eeg_outlet
        else:  # ACCGYRO
            buffer = accgyro_buffer
            outlet = accgyro_outlet

        if len(buffer) == 0:
            return

        from mne_lsl.lsl import local_clock

        # Flatten all buffered samples into a single list
        all_samples = []  # List of (timestamp, data_row) tuples
        for timestamps, chunk in buffer:
            for i, ts in enumerate(timestamps):
                all_samples.append((ts, chunk[i, :]))

        # Sort ALL samples by timestamp (this ensures global monotonicity)
        all_samples.sort(key=lambda x: x[0])

        # Separate timestamps and data after sorting
        combined_timestamps = np.array([s[0] for s in all_samples])
        combined_chunk = np.vstack([s[1] for s in all_samples])

        # Push to LSL
        try:
            outlet.push_chunk(combined_chunk, timestamp=combined_timestamps, pushThrough=True)  # type: ignore[arg-type]
            samples_sent[sensor_type] += len(combined_timestamps)

            # Log to JSON if requested - save exactly what was pushed to LSL
            if lsl_data_log is not None:
                lsl_data_log[sensor_type].append(
                    (combined_timestamps.copy(), combined_chunk.copy())
                )
        except Exception as exc:
            if verbose:
                print(f"LSL push_chunk failed for {sensor_type}: {exc}")

        # Clear buffer and update last push time
        buffer.clear()
        last_push_time[sensor_type] = local_clock()

    def _on_data(_, data: bytearray):
        nonlocal samples_sent, samples_written, device_to_lsl_offset, eeg_buffer, accgyro_buffer, last_push_time
        try:
            # Both EEG and ACC/GYRO data come through EEG characteristic
            message = f"{_ts()}\t{MuseS.EEG_UUID}\t{data.hex()}"
            decoded = parse_message(message)
        except Exception as exc:
            if verbose:
                print(f"Decoding error: {exc}")
            return

        from mne_lsl.lsl import local_clock

        lsl_now = local_clock()

        # Process EEG data
        eeg_data = decoded.get("EEG", np.empty((0, 0)))
        if eeg_data.size > 0 and eeg_data.shape[1] >= 5:  # time + 4 EEG channels
            num_samples = eeg_data.shape[0]

            # Compute device-to-LSL time offset on first packet only
            if device_to_lsl_offset is None:
                device_time_first = eeg_data[0, 0]  # First column is time
                # Calculate offset: LSL_time = device_time + offset
                device_to_lsl_offset = lsl_now - device_time_first
                if verbose:
                    print(
                        f"Initialized time offset: {device_to_lsl_offset:.3f} seconds"
                    )

            # Extract EEG sensor data (exclude time column, keep only first 4 channels): shape (n_samples, 4)
            eeg_chunk = eeg_data[:, 1:5].astype(np.float32)

            # Calculate LSL timestamps for all samples
            device_times = eeg_data[:, 0]
            lsl_timestamps = (device_times + device_to_lsl_offset).tolist()

            # Add to EEG reordering buffer
            eeg_buffer.append((lsl_timestamps, eeg_chunk))

            # Set last_push_time on first data
            if last_push_time["EEG"] is None:
                last_push_time["EEG"] = lsl_now
                if not stream_started.is_set():
                    stream_started.set()

            # Flush buffer if enough time has passed
            if lsl_now - last_push_time["EEG"] >= buffer_duration:
                _flush_buffer("EEG")

        # Process ACCGYRO data
        accgyro_data = decoded.get("ACCGYRO", np.empty((0, 0)))
        if accgyro_data.size > 0:
            num_samples = accgyro_data.shape[0]

            # Compute device-to-LSL time offset on first packet only
            if device_to_lsl_offset is None and num_samples > 0:
                device_time_first = accgyro_data[0, 0]  # First column is time
                # Calculate offset: LSL_time = device_time + offset
                device_to_lsl_offset = lsl_now - device_time_first
                if verbose:
                    print(
                        f"Initialized time offset: {device_to_lsl_offset:.3f} seconds"
                    )

            # Prepare chunk data
            # Extract sensor data (exclude time column): shape (n_samples, 6)
            accgyro_chunk = accgyro_data[:, 1:].astype(np.float32)

            # Calculate LSL timestamps for all samples
            device_times = accgyro_data[:, 0]
            lsl_timestamps = (device_times + device_to_lsl_offset).tolist()

            # Add to ACCGYRO reordering buffer
            accgyro_buffer.append((lsl_timestamps, accgyro_chunk))

            # Set last_push_time on first data
            if last_push_time["ACCGYRO"] is None:
                last_push_time["ACCGYRO"] = lsl_now
                if not stream_started.is_set():
                    stream_started.set()

            # Flush buffer if enough time has passed
            if lsl_now - last_push_time["ACCGYRO"] >= buffer_duration:
                _flush_buffer("ACCGYRO")

    try:
        if verbose:
            print(f"Connecting to {address} ...")

        async with bleak.BleakClient(address, timeout=15.0) as client:
            if verbose:
                print("Connected. Subscribing and configuring ...")

            # Build callbacks dict for all data characteristics
            data_callbacks = {MuseS.EEG_UUID: _on_data}

            # Use shared connection routine
            await MuseS.connect_and_initialize(client, preset, data_callbacks, verbose)

            # Wait briefly for streaming to start before declaring success
            if verbose:
                print("Waiting for streaming to start (up to 3.0s)...")
            try:
                await asyncio.wait_for(stream_started.wait(), timeout=3.0)
                if verbose:
                    print("Streaming started.")
            except asyncio.TimeoutError:
                if verbose:
                    print(
                        "Warning: no data received within 3.0s; will continue up to timeout."
                    )

            if duration:
                if verbose:
                    print(f"Streaming for {duration} seconds...")
                start = time.time()
                try:
                    while time.time() - start < duration:
                        await asyncio.sleep(0.05)
                except asyncio.CancelledError:
                    pass
            else:
                if verbose:
                    print("Streaming indefinitely. Press Ctrl+C to stop.")
                try:
                    while True:
                        await asyncio.sleep(1)
                except asyncio.CancelledError:
                    pass

    except asyncio.CancelledError:
        if verbose:
            print("Stream cancelled.")
    except Exception as exc:
        if verbose:
            print(f"An error occurred: {exc}")
    finally:
        if verbose:
            print("Stopping stream...")

        # Flush any remaining samples in both reordering buffers
        if len(eeg_buffer) > 0:
            if verbose:
                print(f"Flushing {len(eeg_buffer)} buffered EEG packets...")
            _flush_buffer("EEG")

        if len(accgyro_buffer) > 0:
            if verbose:
                print(f"Flushing {len(accgyro_buffer)} buffered ACCGYRO packets...")
            _flush_buffer("ACCGYRO")

        del eeg_outlet
        del accgyro_outlet

        # Write JSON output if requested - save exactly what was sent to LSL
        if lsl_data_log is not None and outfile:
            if verbose:
                total_pushes = len(lsl_data_log["EEG"]) + len(lsl_data_log["ACCGYRO"])
                print(f"Writing {total_pushes} LSL push operations to {outfile}...")
            try:
                # Process EEG data
                eeg_samples = []
                for timestamps, data_chunk in lsl_data_log["EEG"]:
                    for i, ts in enumerate(timestamps):
                        eeg_samples.append((ts, data_chunk[i, :].tolist()))

                eeg_samples.sort(key=lambda x: x[0])
                eeg_timestamps = [s[0] for s in eeg_samples]
                eeg_data_array = [s[1] for s in eeg_samples]

                # Process ACCGYRO data
                accgyro_samples = []
                for timestamps, data_chunk in lsl_data_log["ACCGYRO"]:
                    for i, ts in enumerate(timestamps):
                        accgyro_samples.append((ts, data_chunk[i, :].tolist()))

                accgyro_samples.sort(key=lambda x: x[0])
                accgyro_timestamps = [s[0] for s in accgyro_samples]
                accgyro_data_array = [s[1] for s in accgyro_samples]

                # Build JSON with separate EEG and ACCGYRO data
                json_data = {
                    "EEG": {
                        "lsl_timestamps": eeg_timestamps,
                        "channels": ["TP9", "AF7", "AF8", "TP10"],
                        "data": eeg_data_array,
                        "n_samples": len(eeg_timestamps),
                        "sampling_rate": 256.0,
                        "unit": "microvolts",
                    },
                    "ACCGYRO": {
                        "lsl_timestamps": accgyro_timestamps,
                        "channels": [
                            "ACC_X",
                            "ACC_Y",
                            "ACC_Z",
                            "GYRO_X",
                            "GYRO_Y",
                            "GYRO_Z",
                        ],
                        "data": accgyro_data_array,
                        "n_samples": len(accgyro_timestamps),
                        "sampling_rate": 52.0,
                        "unit": "a.u.",
                    },
                    "note": "LSL data globally sorted by timestamp per sensor type",
                }

                # Ensure output directory exists
                outdir = os.path.dirname(os.path.abspath(outfile))
                if outdir and not os.path.exists(outdir):
                    os.makedirs(outdir, exist_ok=True)

                # Write to file
                with open(outfile, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, indent=2)

                if verbose:
                    print(
                        f"Wrote {len(eeg_timestamps)} EEG samples and {len(accgyro_timestamps)} ACCGYRO samples to {outfile}"
                    )
                    print("File written successfully.")
            except Exception as exc:
                if verbose:
                    print(f"Error writing to file: {exc}")

        if verbose:
            print(
                f"Stream stopped. EEG samples: {samples_sent['EEG']}, ACCGYRO samples: {samples_sent['ACCGYRO']}"
            )


def stream(
    address: str,
    preset: str = "p1041",
    duration: Optional[float] = None,
    outfile: Optional[str] = None,
    verbose: bool = True,
) -> None:
    """
    Stream decoded EEG and accelerometer/gyroscope data over LSL.

    Creates two LSL streams:
    - Muse_EEG: 4 channels (TP9, AF7, AF8, TP10) at 256 Hz
    - Muse_ACCGYRO: 6 channels (ACC_X/Y/Z, GYRO_X/Y/Z) at 52 Hz

    Parameters
    ----------
    address : str
        Device address (e.g., MAC on Windows).
    preset : str
        Preset to send (e.g., p1041 for all channels, p1035 for basic config).
    duration : float, optional
        Optional stream duration in seconds. Omit to stream until interrupted.
    outfile : str, optional
        Optional output JSON file to save decoded EEG and ACC/GYRO samples.
        Omit to only stream without saving.
    verbose : bool
        If True, print verbose output.
    """
    _run(_stream_async(address, preset, duration, outfile, verbose))
