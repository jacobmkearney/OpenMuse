import asyncio
import time
from datetime import datetime, timezone
from typing import Iterable, Optional

import bleak

from .backends import _run

# Muse S Athena service/characteristic UUIDs (from Amused/BrainFlow notes)
MUSE_SERVICE_UUID = "0000fe8d-0000-1000-8000-00805f9b34fb"
CONTROL_CHAR_UUID = "273e0001-4c4d-454d-96be-f03bac821358"
# Data characteristics: combined sensors and secondary
EEG_CHAR_UUID = "273e0013-4c4d-454d-96be-f03bac821358"
OTHER_CHAR_UUID = "273e0014-4c4d-454d-96be-f03bac821358"

# Amused command bytes
COMMANDS = {
    "v6": bytes.fromhex("0376360a"),  # Version request
    "s": bytes.fromhex("02730a"),  # Status
    "h": bytes.fromhex("02680a"),  # Halt
    "p20": bytes.fromhex("047032300a"),
    "p21": bytes.fromhex("047032310a"),
    "p1034": bytes.fromhex("0670313033340a"),
    "p1035": bytes.fromhex("0670313033350a"),
    "dc001": bytes.fromhex("0664633030310a"),  # Start (send twice)
    "L1": bytes.fromhex("034c310a"),
}


async def _record_async(
    address: str,
    timeout: float,
    outfile: str,
    preset: str = "p1035",
    subscribe_chars: Optional[Iterable[str]] = None,
    verbose: bool = True,
    view: bool = False,
    decode_summary: bool = False,
) -> None:
    if subscribe_chars is None:
        subscribe_chars = [EEG_CHAR_UUID, OTHER_CHAR_UUID]

    notified = 0

    def _ts() -> str:
        return datetime.now(timezone.utc).isoformat()

    # Open output file in text mode and append
    f = open(outfile, "a", encoding="utf-8")

    if view and verbose:
        print("Note: live viewer is currently disabled (decoder pending).")
    if decode_summary and verbose:
        print("Note: auto decode summary is disabled (decoder pending).")

    def _callback(uuid: str):
        def inner(_, data: bytearray):
            nonlocal notified
            notified += 1
            # Log timestamp, char UUID, and hex payload
            ts = _ts()
            line = f"{ts}\t{uuid}\t{data.hex()}\n"
            f.write(line)
            # Raw recording only; no decoding or viewing

        return inner

    if verbose:
        print(f"Connecting to {address} ...")

    try:
        async with bleak.BleakClient(address, timeout=15.0) as client:
            if verbose:
                print("Connected. Subscribing and configuring ...")

            # Optionally subscribe to control notifications (best-effort)
            try:
                await client.start_notify(CONTROL_CHAR_UUID, lambda *_: None)
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

            # Command helpers: write without response using known hex commands
            async def write_cmd(key: str):
                data = COMMANDS.get(key)
                if not data:
                    raise ValueError(f"Unknown command: {key}")
                await client.write_gatt_char(CONTROL_CHAR_UUID, data, response=False)

            # Version/status handshake (best-effort)
            try:
                await write_cmd("v6")
                await asyncio.sleep(0.1)
                await write_cmd("s")
                await asyncio.sleep(0.1)
            except Exception as e:
                if verbose:
                    print(f"Warning: version/status failed: {e}")

            # Halt/reset
            try:
                await write_cmd("h")
                await asyncio.sleep(0.1)
            except Exception as e:
                if verbose:
                    print(f"Warning: halt failed: {e}")

            # Preset selection
            if verbose:
                print(f"Sending preset {preset!r} and start commands ...")
            try:
                if preset in COMMANDS:
                    await write_cmd(preset)
                else:
                    # Fallback to ascii command with newline if unknown
                    await client.write_gatt_char(
                        CONTROL_CHAR_UUID,
                        (preset + "\n").encode("ascii"),
                        response=False,
                    )
            except Exception as e:
                if verbose:
                    print(f"Warning: preset {preset!r} failed: {e}")
            await asyncio.sleep(0.1)

            # Start streaming (SEND TWICE), then L1
            await write_cmd("dc001")
            await asyncio.sleep(0.05)
            await write_cmd("dc001")
            await asyncio.sleep(0.1)
            try:
                await write_cmd("L1")
                await asyncio.sleep(0.2)
            except Exception:
                pass

            if verbose:
                print(f"Recording for {timeout} seconds to {outfile} ...")

            # Live viewer disabled for now

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
        # No viewer to close

    if verbose:
        print(f"Done. Wrote {notified} notifications to {outfile}.")
    # Auto decode summary disabled


def record(
    address: str,
    timeout: float = 30.0,
    outfile: str = "muse_record.txt",
    preset: str = "p1035",
    verbose: bool = True,
    view: bool = False,
    decode_summary: bool = False,
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
    - view: (Disabled) Live viewer is currently turned off.
    """
    chars = [EEG_CHAR_UUID, OTHER_CHAR_UUID]
    return _run(
        _record_async(
            address=address,
            timeout=timeout,
            outfile=outfile,
            preset=preset,
            subscribe_chars=chars,
            verbose=verbose,
            view=view,
            decode_summary=decode_summary,
        )
    )
