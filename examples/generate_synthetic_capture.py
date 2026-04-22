"""Generate the richer deterministic synthetic capture used by the example configs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from peekaboo.capture.synthetic import write_synthetic_capture  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "examples" / "captures" / "synthetic-demo.pcap",
        help="Output PCAP path.",
    )
    parser.add_argument(
        "--packet-count",
        type=int,
        default=120,
        help=(
            "Number of synthetic packets to generate. The first 120 packets form one "
            "complete fake traffic story; larger values repeat it with later timestamps."
        ),
    )
    args = parser.parse_args()
    output = write_synthetic_capture(args.output, packet_count=args.packet_count)
    print(f"Wrote synthetic capture to {output}")


if __name__ == "__main__":
    main()
