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


def _build_outlet():

    # Create a single 6-channel outlet for ACC+GYRO data
    labels = ("ACC_X", "ACC_Y", "ACC_Z", "GYRO_X", "GYRO_Y", "GYRO_Z")
    info = StreamInfo(
        name="MuseAccGyro",
        stype="Motion",
        n_channels=6,
        sfreq=52.0,
        dtype="float32",
        source_id="MuseAccGyro",
    )
    desc = info.desc
    desc.append_child_value("manufacturer", "Muse")
    channels = desc.append_child("channels")
    for label in labels:
        channel = channels.append_child("channel")
        channel.append_child_value("label", label)
        channel.append_child_value("unit", "a.u.")

    return StreamOutlet(info, chunk_size=1)


async def _stream_async(
    address: str,
    preset: str,
    duration: Optional[float],
    outfile: Optional[str],
    verbose: bool,
) -> None:
    stream_started = asyncio.Event()
    outlet = _build_outlet()
    samples_sent = 0
    samples_written = 0

    # Compute device-to-LSL time offset once at the start
    # Will be updated with the first data packet
    device_to_lsl_offset = None

    # Reordering buffer to handle out-of-order BLE messages
    # Stores (lsl_timestamp, chunk_data) tuples
    reorder_buffer = []
    buffer_duration = 0.2  # Hold samples for 200ms to allow reordering (BLE can be delayed up to ~40ms)
    last_push_time = None

    def _ts() -> str:
        return datetime.now(timezone.utc).isoformat()

    # Collect LSL data for JSON output (exactly what gets pushed to LSL)
    # This will be populated during _flush_buffer() calls
    lsl_data_log = [] if outfile else None  # List of (timestamps, data) after sorting

    def _flush_buffer():
        """Flush reordering buffer: sort and push samples to LSL."""
        nonlocal samples_sent, reorder_buffer, last_push_time

        if len(reorder_buffer) == 0:
            return

        from mne_lsl.lsl import local_clock

        # Flatten all buffered samples into a single list
        all_samples = []  # List of (timestamp, data_row) tuples
        for timestamps, chunk in reorder_buffer:
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
            samples_sent += len(combined_timestamps)

            # Log to JSON if requested - save exactly what was pushed to LSL
            if lsl_data_log is not None:
                lsl_data_log.append((combined_timestamps.copy(), combined_chunk.copy()))
        except Exception as exc:
            if verbose:
                print(f"LSL push_chunk failed: {exc}")

        # Clear buffer and update last push time
        reorder_buffer.clear()
        last_push_time = local_clock()

    def _on_data(_, data: bytearray):
        nonlocal samples_sent, samples_written, device_to_lsl_offset, reorder_buffer, last_push_time
        try:
            # ACC/GYRO data comes through EEG characteristic, not OTHER
            message = f"{_ts()}\t{MuseS.EEG_UUID}\t{data.hex()}"
            decoded = parse_message(message)
        except Exception as exc:
            if verbose:
                print(f"Decoding error: {exc}")
            return

        # Get ACCGYRO data (numpy array with shape (n_samples, 7): time + 6 channels)
        accgyro_data = decoded.get("ACCGYRO", np.empty((0, 0)))

        if accgyro_data.size == 0:
            return

        num_samples = accgyro_data.shape[0]

        # Get LSL timestamp for this BLE packet arrival
        from mne_lsl.lsl import local_clock

        lsl_now = local_clock()

        # Compute device-to-LSL time offset on first packet only
        if device_to_lsl_offset is None and num_samples > 0:
            device_time_first = accgyro_data[0, 0]  # First column is time
            # Calculate offset: LSL_time = device_time + offset
            device_to_lsl_offset = lsl_now - device_time_first
            if verbose:
                print(f"Initialized time offset: {device_to_lsl_offset:.3f} seconds")

        # Prepare chunk data
        # Extract sensor data (exclude time column): shape (n_samples, 6)
        chunk = accgyro_data[:, 1:].astype(np.float32)

        # Calculate LSL timestamps for all samples
        device_times = accgyro_data[:, 0]
        lsl_timestamps = (device_times + device_to_lsl_offset).tolist()

        # Add to reordering buffer
        reorder_buffer.append((lsl_timestamps, chunk))

        # Set last_push_time on first data
        if last_push_time is None:
            last_push_time = lsl_now
            if not stream_started.is_set():
                stream_started.set()

        # Flush buffer if enough time has passed
        if lsl_now - last_push_time >= buffer_duration:
            _flush_buffer()

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

        # Flush any remaining samples in the reordering buffer
        if len(reorder_buffer) > 0:
            if verbose:
                print(f"Flushing {len(reorder_buffer)} buffered packets...")
            _flush_buffer()

        del outlet

        # Write JSON output if requested - save exactly what was sent to LSL
        if lsl_data_log is not None and outfile:
            if verbose:
                print(
                    f"Writing {len(lsl_data_log)} LSL push operations to {outfile}..."
                )
            try:
                # Flatten all LSL pushes and sort globally
                # (Each flush is sorted internally, but flushes may overlap in time)
                all_samples = []
                for timestamps, data_chunk in lsl_data_log:
                    for i, ts in enumerate(timestamps):
                        all_samples.append((ts, data_chunk[i, :]))

                # Sort ALL samples by timestamp to ensure global monotonicity
                all_samples.sort(key=lambda x: x[0])

                # Separate timestamps and data
                all_timestamps = [s[0] for s in all_samples]
                all_data_array = np.vstack([s[1] for s in all_samples])

                # Build minimal JSON: timestamps + channel data
                # This is exactly what LSL received, globally sorted
                json_data = {
                    "lsl_timestamps": all_timestamps,
                    "channels": [
                        "ACC_X",
                        "ACC_Y",
                        "ACC_Z",
                        "GYRO_X",
                        "GYRO_Y",
                        "GYRO_Z",
                    ],
                    "data": all_data_array.tolist(),  # Shape: (n_samples, 6)
                    "n_samples": len(all_timestamps),
                    "note": "LSL data globally sorted by timestamp",
                }

                # Ensure output directory exists
                outdir = os.path.dirname(os.path.abspath(outfile))
                if outdir and not os.path.exists(outdir):
                    os.makedirs(outdir, exist_ok=True)

                # Write to file
                with open(outfile, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, indent=2)

                if verbose:
                    print(f"Wrote {len(all_timestamps)} samples to {outfile}")
                    print("File written successfully.")
            except Exception as exc:
                if verbose:
                    print(f"Error writing to file: {exc}")

        if verbose:
            print(f"Stream stopped. Total samples sent: {samples_sent}")


def stream(
    address: str,
    preset: str = "p1041",
    duration: Optional[float] = None,
    outfile: Optional[str] = None,
    verbose: bool = True,
) -> None:
    """
    Stream decoded accelerometer and gyroscope data over LSL.

    Parameters
    ----------
    address : str
        Device address (e.g., MAC on Windows).
    preset : str
        Preset to send (e.g., p1035 or p21).
    duration : float, optional
        Optional stream duration in seconds. Omit to stream until interrupted.
    outfile : str, optional
        Optional output file to save decoded ACC/GYRO samples. Omit to only stream.
    verbose : bool
        If True, print verbose output.
    """
    _run(_stream_async(address, preset, duration, outfile, verbose))
