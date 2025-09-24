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
        "record", help="Connect and record raw packets to a text file"
    )
    p_rec.add_argument(
        "--address", required=True, help="Device address (e.g., MAC on Windows)"
    )
    p_rec.add_argument(
        "--timeout",
        "-t",
        type=float,
        default=30.0,
        help="Recording timeout in seconds (default: 30)",
    )
    p_rec.add_argument(
        "--outfile", "-o", default="muse_record.txt", help="Output text file path"
    )
    p_rec.add_argument(
        "--preset", default="p1035", help="Preset to send (e.g., p1035 or p21)"
    )

    def handle_record(ns):
        if ns.timeout <= 0:
            parser.error("--timeout must be positive")
        record(
            address=ns.address,
            timeout=ns.timeout,
            outfile=ns.outfile,
            preset=ns.preset,
            verbose=True,
        )
        return 0

    p_rec.set_defaults(func=handle_record)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("Interrupted.")
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
