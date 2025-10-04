import asyncio
import json
import os
import time
from datetime import datetime, timezone
from typing import Optional

import bleak
import numpy as np

from .backends import _run
from .decode import decode_message
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
            decoded = decode_message(message)
        except Exception as exc:
            if verbose:
                print(f"Decoding error: {exc}")
            return

        if not decoded:
            return

        # Combine ACC and GYRO data into 6-channel samples
        acc_entries = decoded.get("ACC", [])
        gyro_entries = decoded.get("GYRO", [])

        # ACC and GYRO should have the same number of samples
        num_samples = min(len(acc_entries), len(gyro_entries))
        for i in range(num_samples):
            acc_x = acc_entries[i].get("ACC_X", 0.0)
            acc_y = acc_entries[i].get("ACC_Y", 0.0)
            acc_z = acc_entries[i].get("ACC_Z", 0.0)
            gyro_x = gyro_entries[i].get("GYRO_X", 0.0)
            gyro_y = gyro_entries[i].get("GYRO_Y", 0.0)
            gyro_z = gyro_entries[i].get("GYRO_Z", 0.0)

            sample = np.array(
                [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z],
                dtype=np.float32,
            )

            # Push to LSL and capture timestamp
            lsl_timestamp = None
            try:
                # Push sample and capture the LSL timestamp
                from mne_lsl.lsl import local_clock

                lsl_timestamp = local_clock()
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
                    ts = acc_entries[i].get("time")
                    # ts is already a float (seconds since epoch) from decode_message
                    ts_float = ts if ts is not None else time.time()
                    json_data["ACC"]["time"].append(ts_float)
                    json_data["ACC"]["time_lsl"].append(
                        lsl_timestamp if lsl_timestamp else 0.0
                    )
                    json_data["ACC"]["ACC_X"].append(acc_x)
                    json_data["ACC"]["ACC_Y"].append(acc_y)
                    json_data["ACC"]["ACC_Z"].append(acc_z)
                    json_data["GYRO"]["time"].append(ts_float)
                    json_data["GYRO"]["time_lsl"].append(
                        lsl_timestamp if lsl_timestamp else 0.0
                    )
                    json_data["GYRO"]["GYRO_X"].append(gyro_x)
                    json_data["GYRO"]["GYRO_Y"].append(gyro_y)
                    json_data["GYRO"]["GYRO_Z"].append(gyro_z)
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

            # ACC/GYRO data is multiplexed with EEG in the EEG characteristic
            # Subscribe to both characteristics to maintain connection
            def _on_other(_, data: bytearray):
                # OTHER characteristic may have additional data, ignore for now
                pass

            data_callbacks = {
                MuseS.EEG_UUID: _on_data,
                MuseS.OTHER_UUID: _on_other,
            }
            await MuseS.connect_and_initialize(client, preset, data_callbacks, verbose)

            if verbose:
                print("Waiting for motion packets (up to 3.0s)...")
            try:
                await asyncio.wait_for(stream_started.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                if verbose:
                    print("Warning: no motion data received within 3.0s.")

            if duration is None:
                if verbose:
                    print("Streaming acc/gyro indefinitely. Press Ctrl+C to stop.")
            else:
                if duration <= 0:
                    raise ValueError("duration must be positive")
                if verbose:
                    print(f"Streaming acc/gyro for {duration:.1f} seconds ...")

            stop_at = None if duration is None else time.monotonic() + duration
            try:
                while True:
                    if stop_at is not None and time.monotonic() >= stop_at:
                        break
                    await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                pass
            finally:
                # Try to stop streaming gracefully
                await MuseS.stop_streaming(client, verbose)

            # Unsubscribe from all characteristics
            try:
                await client.stop_notify(MuseS.EEG_UUID)
            except Exception:
                pass
            try:
                await client.stop_notify(MuseS.OTHER_UUID)
            except Exception:
                pass
            try:
                await client.stop_notify(MuseS.CONTROL_UUID)
            except Exception:
                pass

    finally:
        # Write JSON data to file if accumulated
        if json_data is not None and outfile:
            try:
                outdir = os.path.dirname(os.path.abspath(outfile))
                if outdir and not os.path.exists(outdir):
                    os.makedirs(outdir, exist_ok=True)
            except Exception:
                pass

            try:
                with open(outfile, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, indent=2)
                if verbose:
                    print(f"Wrote {samples_written} samples to {outfile}")
            except Exception as exc:
                if verbose:
                    print(f"Failed to write JSON file: {exc}")

        if verbose:
            msg = f"Stopped streaming. Sent {samples_sent} ACC+GYRO samples."
            if json_data is not None:
                msg += f" Saved {samples_written} samples."
            print(msg)


def stream(
    address: str,
    preset: str = "p1035",
    duration: Optional[float] = None,
    outfile: Optional[str] = None,
    verbose: bool = True,
) -> None:
    """
    Connect to a Muse, decode acc/gyro data, and stream over LSL using mne-lsl.

    Creates a single LSL outlet named "MuseAccGyro" with 6 channels:
    ACC_X, ACC_Y, ACC_Z, GYRO_X, GYRO_Y, GYRO_Z at 52 Hz nominal rate.

    Parameters
    ----------
    address : str
        Device MAC/identifier for the Muse headset.
    preset : str
        Preset token to send before streaming (default ``"p1035"``).
    duration : float | None
        Optional stream length in seconds. ``None`` streams until interrupted.
    outfile : str | None
        Optional file path to save decoded ACC/GYRO samples. ``None`` streams only.
        File format: JSON with dict structure:
            {"ACC": {"time": [...], "time_lsl": [...], "ACC_X": [...], "ACC_Y": [...], "ACC_Z": [...]},
             "GYRO": {"time": [...], "time_lsl": [...], "GYRO_X": [...], "GYRO_Y": [...], "GYRO_Z": [...]}}
        where 'time' is device timestamp and 'time_lsl' is LSL timestamp for synchronization.
    verbose : bool
        Whether to print progress messages.
    """
    if not address:
        raise ValueError("address must be a non-empty string")

    return _run(
        _stream_async(
            address=address,
            preset=preset,
            duration=duration,
            outfile=outfile,
            verbose=verbose,
        )
    )
