import asyncio
import json
import re
import os
import time
from datetime import datetime, timezone
from typing import Iterable, Optional

import bleak

from .backends import _run

# Muse S Athena characteristic UUIDs
CONTROL_CHAR_UUID = "273e0001-4c4d-454d-96be-f03bac821358"
EEG_CHAR_UUID = "273e0013-4c4d-454d-96be-f03bac821358"
OTHER_CHAR_UUID = "273e0014-4c4d-454d-96be-f03bac821358"


def _encode_command(token: str) -> bytes:
    """
    Encode a command token using the Muse/Amused convention:
    [length byte] + ASCII(token) + "\n"

    The length includes the trailing newline and must be <= 255.
    """
    if not isinstance(token, str) or not token:
        raise ValueError("command token must be a non-empty string")
    try:
        payload = (token + "\n").encode("ascii")
    except UnicodeEncodeError as e:
        raise ValueError("command token must be ASCII") from e
    if len(payload) > 255:
        raise ValueError("command too long (max 254 chars plus newline)")
    return bytes([len(payload)]) + payload


async def _record_async(
    address: str,
    timeout: float,
    outfile: str,
    preset: str = "p1035",
    subscribe_chars: Optional[Iterable[str]] = None,
    verbose: bool = True,
) -> None:
    if subscribe_chars is None:
        subscribe_chars = [EEG_CHAR_UUID, OTHER_CHAR_UUID]

    notified = 0
    stream_started = asyncio.Event()
    device_info_logged = {"fw": False, "bp": False, "text": False}
    control_buffer = ""

    def _ts() -> str:
        return datetime.now(timezone.utc).isoformat()

    # Ensure output directory exists
    try:
        outdir = os.path.dirname(os.path.abspath(outfile))
        if outdir and not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)
    except Exception:
        pass

    # Open output file in text mode and append
    f = open(outfile, "a", encoding="utf-8")

    # Raw recording only; no live viewer or decoder

    def _callback(uuid: str):
        def inner(_, data: bytearray):
            nonlocal notified
            notified += 1
            # Log timestamp, char UUID, and hex payload
            ts = _ts()
            line = f"{ts}\t{uuid}\t{data.hex()}\n"
            f.write(line)
            # Raw recording only; no decoding or viewing
            if not stream_started.is_set():
                stream_started.set()

        return inner

    if verbose:
        print(f"Connecting to {address} ...")

    try:
        async with bleak.BleakClient(address, timeout=15.0) as client:
            if verbose:
                print("Connected. Subscribing and configuring ...")

            # Optionally subscribe to control notifications (best-effort)
            try:

                def _on_control(_, data: bytearray):
                    # Try to decode and extract device info from control notifications
                    try:
                        text = data.decode("utf-8", errors="ignore")
                    except Exception:
                        return

                    nonlocal control_buffer
                    # Accumulate to handle fragmented JSON across notifications
                    control_buffer = (control_buffer + text)[-4096:]

                    # Attempt JSON parse first (preferred)
                    try:
                        if "{" in control_buffer and "}" in control_buffer:
                            start = control_buffer.index("{")
                            end = control_buffer.rindex("}") + 1
                            payload = control_buffer[start:end]
                            info = json.loads(payload)
                            if verbose:
                                if not device_info_logged["fw"] and "fw" in info:
                                    print(f"Firmware: {info['fw']}")
                                    device_info_logged["fw"] = True
                                if not device_info_logged["bp"] and "bp" in info:
                                    bp_val = info["bp"]
                                    try:
                                        bp_int = int(bp_val)
                                        print(f"Battery: {bp_int}%")
                                    except Exception:
                                        print(f"Battery: {bp_val}%")
                                    device_info_logged["bp"] = True
                    except Exception:
                        pass

                    # Heuristics in plain text if JSON not present/parsed
                    if verbose:
                        if not device_info_logged["bp"]:
                            m = re.search(
                                r"(bp|battery|batt|charge)[^0-9]{0,10}(\d{1,3})",
                                text,
                                re.I,
                            )
                            if m:
                                val = int(m.group(2))
                                if 0 <= val <= 100:
                                    print(f"Battery: {val}%")
                                    device_info_logged["bp"] = True
                        if not device_info_logged["fw"] and re.search(
                            r"\bfw\b", text, re.I
                        ):
                            mfw = re.search(
                                r"\bfw\b\s*[:=\-]?\s*\"?([\w\.-]+)\"?", text, re.I
                            )
                            if mfw:
                                print(f"Firmware: {mfw.group(1)}")
                                device_info_logged["fw"] = True
                        # Fallback: print first control text once to aid debugging
                        if not device_info_logged["text"]:
                            snippet = text.strip().replace("\n", " ")
                            if snippet:
                                print(f"Control: {snippet[:120]}")
                                device_info_logged["text"] = True

                await client.start_notify(CONTROL_CHAR_UUID, _on_control)
                if verbose:
                    print(f"Subscribed to notifications on {CONTROL_CHAR_UUID}")
            except Exception as e:
                if verbose:
                    print(
                        f"Warning: could not subscribe control {CONTROL_CHAR_UUID}: {e}"
                    )

            # Subscribe to desired data characteristics
            for cuuid in subscribe_chars:
                try:
                    await client.start_notify(cuuid, _callback(cuuid))
                    if verbose:
                        print(f"Subscribed to notifications on {cuuid}")
                except Exception as e:
                    if verbose:
                        print(f"Warning: could not subscribe {cuuid}: {e}")

            # Command helper: length-prefixed ASCII encoder used for all tokens
            async def write_cmd(token: str):
                data = _encode_command(token)
                await client.write_gatt_char(CONTROL_CHAR_UUID, data, response=False)

            # Version/status handshake (best-effort)
            try:
                await write_cmd("v6")
                await asyncio.sleep(0.2)
                await write_cmd("s")
                await asyncio.sleep(0.2)
            except Exception as e:
                if verbose:
                    print(f"Warning: version/status failed: {e}")

            # Halt/reset
            try:
                await write_cmd("h")
                await asyncio.sleep(0.2)
            except Exception as e:
                if verbose:
                    print(f"Warning: halt failed: {e}")

            # Preset selection
            if verbose:
                print(f"Sending preset {preset!r} and start commands ...")
            try:
                await write_cmd(preset)
            except Exception as e:
                if verbose:
                    print(f"Warning: preset {preset!r} failed: {e}")
            await asyncio.sleep(0.2)
            # Query status again after preset change
            try:
                await write_cmd("s")
                await asyncio.sleep(0.2)
            except Exception:
                pass

            # Start streaming (SEND TWICE), then L1
            await write_cmd("dc001")
            await asyncio.sleep(0.05)
            await write_cmd("dc001")
            await asyncio.sleep(0.1)
            try:
                await write_cmd("L1")
                await asyncio.sleep(0.3)
            except Exception:
                pass
            # Try status once more after start
            try:
                await write_cmd("s")
                await asyncio.sleep(0.2)
            except Exception:
                pass

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

            if verbose:
                print(f"Recording for {timeout} seconds to {outfile} ...")

            start = time.time()
            try:
                while time.time() - start < timeout:
                    await asyncio.sleep(0.05)
            except asyncio.CancelledError:
                pass
            finally:
                # Try to stop streaming gracefully (halt)
                try:
                    await write_cmd("h")
                except Exception:
                    pass

            # Unsubscribe
            for cuuid in subscribe_chars:
                try:
                    await client.stop_notify(cuuid)
                except Exception:
                    pass
            # Also try to stop control notify
            try:
                await client.stop_notify(CONTROL_CHAR_UUID)
            except Exception:
                pass

    finally:
        try:
            f.flush()
            f.close()
        except Exception:
            pass
    if verbose:
        print(f"Done. Wrote {notified} notifications to {outfile}.")


def record(
    address: str,
    timeout: float = 30.0,
    outfile: str = "muse_record.txt",
    preset: str = "p1035",
    verbose: bool = True,
) -> None:
    """
    Connect to a Muse device, stream notifications, and append raw packets to a text file.

    Each line written: ISO8601 UTC timestamp, characteristic UUID, hex payload.

    Parameters
    - address: Device MAC address (Windows) or identifier (platform-dependent).
    - timeout: Recording duration in seconds.
    - outfile: Path to output text file.
    - preset: Preset string to send (e.g., "p1035" or "p21").
    - verbose: Print progress messages.
    """
    if not address:
        raise ValueError("address must be a non-empty string")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if not isinstance(outfile, str) or not outfile:
        raise ValueError("outfile must be a non-empty path string")

    chars = [EEG_CHAR_UUID, OTHER_CHAR_UUID]
    return _run(
        _record_async(
            address=address,
            timeout=timeout,
            outfile=outfile,
            preset=preset,
            subscribe_chars=chars,
            verbose=verbose,
        )
    )
