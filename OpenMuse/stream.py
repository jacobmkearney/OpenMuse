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

    def _ts() -> str:
        return datetime.now(timezone.utc).isoformat()

    # Prepare data structures for JSON output if requested
    json_data = None
    if outfile:
        json_data = {
            "ACC": {"time": [], "time_lsl": [], "ACC_X": [], "ACC_Y": [], "ACC_Z": []},
            "GYRO": {
                "time": [],
                "time_lsl": [],
                "GYRO_X": [],
                "GYRO_Y": [],
                "GYRO_Z": [],
            },
        }

    def _on_data(_, data: bytearray):
        nonlocal samples_sent, samples_written
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

        # Get LSL timestamp for this BLE packet to establish device-to-LSL time mapping
        from mne_lsl.lsl import local_clock

        lsl_now = local_clock()

        # Use the first sample's device timestamp to compute the offset
        # This maps device time (device uptime in seconds) to LSL time
        device_to_lsl_offset = 0.0  # Initialize default offset
        if num_samples > 0:
            device_time_first = accgyro_data[0, 0]  # First column is time
            # Calculate offset: LSL_time = device_time + offset
            device_to_lsl_offset = lsl_now - device_time_first

        for i in range(num_samples):
            # Extract data: columns are [time, ACC_X, ACC_Y, ACC_Z, GYRO_X, GYRO_Y, GYRO_Z]
            device_time = accgyro_data[i, 0]
            acc_x = accgyro_data[i, 1]
            acc_y = accgyro_data[i, 2]
            acc_z = accgyro_data[i, 3]
            gyro_x = accgyro_data[i, 4]
            gyro_y = accgyro_data[i, 5]
            gyro_z = accgyro_data[i, 6]

            sample = np.array(
                [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z],
                dtype=np.float32,
            )

            # Use actual device timestamp for LSL timestamp
            try:
                # Convert device time to LSL time using the offset
                lsl_timestamp = device_time + device_to_lsl_offset

                outlet.push_sample(sample, lsl_timestamp)
                samples_sent += 1
                if not stream_started.is_set():
                    stream_started.set()
            except Exception as exc:
                if verbose:
                    print(f"LSL push failed: {exc}")
                break

            # Accumulate data for JSON if requested
            if json_data is not None:
                try:
                    json_data["ACC"]["time"].append(float(device_time))
                    json_data["ACC"]["time_lsl"].append(float(lsl_timestamp))
                    json_data["ACC"]["ACC_X"].append(float(acc_x))
                    json_data["ACC"]["ACC_Y"].append(float(acc_y))
                    json_data["ACC"]["ACC_Z"].append(float(acc_z))
                    json_data["GYRO"]["time"].append(float(device_time))
                    json_data["GYRO"]["time_lsl"].append(float(lsl_timestamp))
                    json_data["GYRO"]["GYRO_X"].append(float(gyro_x))
                    json_data["GYRO"]["GYRO_Y"].append(float(gyro_y))
                    json_data["GYRO"]["GYRO_Z"].append(float(gyro_z))
                    samples_written += 1
                except Exception as exc:
                    if verbose:
                        print(f"Data accumulation failed: {exc}")

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
        del outlet
        if json_data and outfile:
            if verbose:
                print(f"Writing {samples_written} samples to {outfile}...")
            try:
                # Ensure output directory exists
                outdir = os.path.dirname(os.path.abspath(outfile))
                if outdir and not os.path.exists(outdir):
                    os.makedirs(outdir, exist_ok=True)
                with open(outfile, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, indent=2)
                if verbose:
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
