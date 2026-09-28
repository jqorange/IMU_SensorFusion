"""Command-line interface for data preparation and fusion."""

from __future__ import annotations

import argparse
from pathlib import Path

from imu_fusion.config import FusionConfig
from imu_fusion.io import read_aligned_imu, write_aligned_imu
from imu_fusion.pipeline import fuse_aligned_imu
from imu_fusion.prepare import prepare_aligned_segment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="imu-fusion")
    subparsers = parser.add_subparsers(dest="command", required=True)
    fuse = subparsers.add_parser("fuse", help="Fuse a start-aligned 9-axis file.")
    fuse.add_argument("input", type=Path)
    fuse.add_argument("output", type=Path)

    prepare = subparsers.add_parser("prepare", help="Prepare an aligned raw segment.")
    prepare.add_argument("analog", type=Path)
    prepare.add_argument("output", type=Path)
    prepare.add_argument("--start-offset", type=float, required=True)
    prepare.add_argument("--duration", type=float, required=True)
    return parser


def main() -> None:
    """Run the project CLI."""
    args = _parser().parse_args()
    config = FusionConfig()
    if args.command == "prepare":
        values = prepare_aligned_segment(
            args.analog, config, args.start_offset, args.duration
        )
        write_aligned_imu(args.output, values, config.sample_rate_hz)
        print(f"Prepared {len(values)} samples: {args.output}")
        return
    values, _ = read_aligned_imu(args.input)
    result = fuse_aligned_imu(values, config)
    result.write(args.output)
    print(f"Fused {len(values)} samples: {args.output}")


if __name__ == "__main__":
    main()
