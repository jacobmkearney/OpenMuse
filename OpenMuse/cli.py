import argparse
import sys

from .find import find_devices, resolve_address
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
        "record", help="Connect and record raw packets to a text file"
    )
    p_rec.add_argument(
        "--address", required=False, help="Device address (e.g., MAC on Windows). Omit to autodiscover."
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
        "--preset", default="p1041", help="Preset to send (by default, p1041)"
    )

    def handle_record(ns):
        if ns.duration <= 0:
            parser.error("--duration must be positive")

        address = ns.address
        if not address:
            address = resolve_address()
            print(f"Autodiscovered device: {address}")

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
        help="Stream decoded EEG and accelerometer/gyroscope data over LSL",
    )
    p_stream.add_argument(
        "--address",
        required=False,
        help="Device address (e.g., MAC on Windows). Omit to autodiscover.",
    )
    p_stream.add_argument(
        "--preset",
        default="p1041",
        help="Preset to send (default: p1041 for all channels including EEG)",
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
        help="Optional output JSON file to save decoded EEG and ACC/GYRO samples. Omit to only stream.",
    )

    def handle_stream(ns):
        from .stream import stream

        if ns.duration is not None and ns.duration <= 0:
            parser.error("--duration must be positive when provided")
        address = ns.address

        if not address:
            address = resolve_address()
            print(f"Autodiscovered device: {address}")

        stream(
            address=address,
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
        help="Visualize EEG and ACC/GYRO data from LSL streams in real-time",
    )
    p_view.add_argument(
        "--stream-name",
        default=None,
        help="Name of specific LSL stream to visualize (default: None = show all available streams: Muse_EEG + Muse_ACCGYRO)",
    )
    p_view.add_argument(
        "--groups",
        default=None,
        help=(
            "Comma-separated channel groups to display (case-insensitive). "
            "Options: EEG, OPTICS, ACC, GYRO, ACCGYRO. Default: all"
        ),
    )
    p_view.add_argument(
        "--channels",
        default=None,
        help=(
            "Comma-separated channel names to include (e.g., EEG_TP9,ACC_X). "
            "Matches are case-insensitive. Default: include all in selected groups"
        ),
    )
    p_view.add_argument(
        "--scaling",
        default="group",
        choices=["group", "channel", "global"],
        help=(
            "Y-axis scaling mode: 'group' (zoom all in group under mouse), "
            "'channel' (zoom only channel under mouse), or 'global' (zoom all channels). "
            "Default: group"
        ),
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

        # Parse comma-separated options into lists (None if not provided)
        groups = (
            [s.strip() for s in ns.groups.split(",") if s.strip()]
            if ns.groups
            else None
        )
        channels = (
            [s.strip() for s in ns.channels.split(",") if s.strip()]
            if ns.channels
            else None
        )

        view(
            stream_name=ns.stream_name,
            duration=ns.duration,
            window_size=ns.window,
            groups=groups,
            channels=channels,
            scaling_mode=ns.scaling,
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
    except ValueError:
        # Discovery raised a user-facing error; upstream already printed details
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
