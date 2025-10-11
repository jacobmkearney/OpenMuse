import argparse
import sys

from .find import find_devices
from .record import record


def _add_find_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-t",
        "--timeout",
        type=int,
        default=10,
        help="Scan timeout in seconds (default: 10)",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(prog="MuseLSL3", description="MuseLSL3 utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # find subcommand
    p_find = subparsers.add_parser("find", help="Scan for Muse devices")
    _add_find_args(p_find)

    def handle_find(ns):
        find_devices(timeout=ns.timeout, verbose=True)
        return 0

    p_find.set_defaults(func=handle_find)

    # record subcommand
    p_rec = subparsers.add_parser(
        "record", help="Connect and record raw packets to a file (text or .bin)"
    )
    p_rec.add_argument(
        "--address", required=False, help="Optional device address. omit to auto-discover"
    )
    p_rec.add_argument(
        "--duration",
        "-d",
        type=float,
        default=30.0,
        help="Recording duration in seconds (default: 30)",
    )
    p_rec.add_argument(
        "--outfile", "-o", default="muse_record.txt", help="Output text file path"
    )
    p_rec.add_argument(
        "--preset", default="p1035", help="Preset to send (e.g., p1035 or p21)"
    )

    def handle_record(ns):
        if ns.duration <= 0:
            parser.error("--duration must be positive")
        # Auto-discover address if not provided
        address = ns.address
        if not address:
            found = find_devices(timeout=10, verbose=True)
            if not found:
                parser.error("No Muse devices discovered. Ensure headset is on and in range.")
            address = found[0]["address"]
            print(f"Using discovered device address: {address}")
        record(
            address=address,
            duration=ns.duration,
            outfile=ns.outfile,
            preset=ns.preset,
            verbose=True,
        )
        return 0

    p_rec.set_defaults(func=handle_record)

    # stream subcommand
    p_stream = subparsers.add_parser(
        "stream",
        help="Stream decoded accelerometer and gyroscope data over LSL",
    )
    p_stream.add_argument(
        "--address",
        required=True,
        help="Device address (e.g., MAC on Windows)",
    )
    p_stream.add_argument(
        "--preset",
        default="p1035",
        help="Preset to send (e.g., p1035 or p21)",
    )
    p_stream.add_argument(
        "--duration",
        "-d",
        type=float,
        default=None,
        help="Optional stream duration in seconds. Omit to stream until interrupted.",
    )
    p_stream.add_argument(
        "--outfile",
        "-o",
        default=None,
        help="Optional output file to save decoded ACC/GYRO samples. Omit to only stream.",
    )

    def handle_stream(ns):
        from .stream import stream

        if ns.duration is not None and ns.duration <= 0:
            parser.error("--duration must be positive when provided")
        stream(
            address=ns.address,
            preset=ns.preset,
            duration=ns.duration,
            outfile=ns.outfile,
            verbose=True,
        )
        return 0

    p_stream.set_defaults(func=handle_stream)

    # view subcommand
    p_view = subparsers.add_parser(
        "view",
        help="Visualize ACC/GYRO data from an LSL stream in real-time",
    )
    p_view.add_argument(
        "--stream-name",
        default="MuseAccGyro",
        help="Name of the LSL stream to visualize (default: MuseAccGyro)",
    )
    p_view.add_argument(
        "--window",
        "-w",
        type=float,
        default=10.0,
        help="Time window to display in seconds (default: 10.0)",
    )
    p_view.add_argument(
        "--duration",
        "-d",
        type=float,
        default=None,
        help="Optional viewing duration in seconds. Omit to view until window closed.",
    )

    def handle_view(ns):
        from .view import view

        if ns.window <= 0:
            parser.error("--window must be positive")
        if ns.duration is not None and ns.duration <= 0:
            parser.error("--duration must be positive when provided")
        view(
            stream_name=ns.stream_name,
            duration=ns.duration,
            window_size=ns.window,
            verbose=True,
        )
        return 0

    p_view.set_defaults(func=handle_view)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("Interrupted.")
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
