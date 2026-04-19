"""Generate the synthetic capture used by the example config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from peekabo.capture.synthetic import write_synthetic_capture  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "examples" / "captures" / "synthetic-demo.pcap",
        help="Output PCAP path.",
    )
    args = parser.parse_args()
    output = write_synthetic_capture(args.output)
    print(f"Wrote synthetic capture to {output}")


if __name__ == "__main__":
    main()
