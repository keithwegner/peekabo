"""Capture preflight diagnostics."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from peekaboo.capture.sources import expand_input_paths, iter_packet_records
from peekaboo.labeling.targets import TargetRegistry
from peekaboo.parsing.dot11 import frame_family
from peekaboo.parsing.records import PacketRecord

RADIOTAP_FIELDS = ("data_rate", "channel_frequency", "rssi_or_ssi")
FRAME_FAMILIES = ("management", "control", "data", "extension", "unknown")


def inspect_capture_paths(
    paths: Iterable[str | Path],
    *,
    registry: TargetRegistry | None = None,
    max_packets: int | None = None,
) -> dict[str, Any]:
    """Inspect PCAP inputs without writing normalized packet datasets."""
    capture_paths = expand_input_paths(paths)
    return summarize_packet_records(
        iter_packet_records(capture_paths),
        capture_paths=capture_paths,
        registry=registry,
        max_packets=max_packets,
    )


def summarize_packet_records(
    records: Iterable[PacketRecord | dict[str, Any]],
    *,
    capture_paths: Iterable[str | Path],
    registry: TargetRegistry | None = None,
    max_packets: int | None = None,
) -> dict[str, Any]:
    capture_files = [str(path) for path in capture_paths]
    packets_scanned = 0
    parse_successes = 0
    parse_failures = 0
    protected_frame_count = 0
    field_present = Counter()
    frame_counts = Counter({family: 0 for family in FRAME_FAMILIES})
    channel_frequencies: set[int] = set()
    source_macs: set[str] = set()
    destination_macs: set[str] = set()
    source_mac_counts = Counter()
    destination_mac_counts = Counter()
    target_matches = Counter()

    for record in records:
        if max_packets is not None and packets_scanned >= max_packets:
            break
        row = record.to_dict() if isinstance(record, PacketRecord) else dict(record)
        packets_scanned += 1

        if row.get("parse_ok", True):
            parse_successes += 1
        else:
            parse_failures += 1

        family = frame_family(row.get("frame_type")) or "unknown"
        frame_counts[family] += 1

        if row.get("protected"):
            protected_frame_count += 1
        if row.get("data_rate") is not None:
            field_present["data_rate"] += 1
        if row.get("channel_frequency") is not None:
            field_present["channel_frequency"] += 1
            channel_frequencies.add(int(row["channel_frequency"]))
        if row.get("rssi") is not None or row.get("ssi") is not None:
            field_present["rssi_or_ssi"] += 1

        source_mac = row.get("source_mac")
        destination_mac = row.get("destination_mac")
        if source_mac:
            source_mac_text = str(source_mac)
            source_macs.add(source_mac_text)
            source_mac_counts[source_mac_text] += 1
        if destination_mac:
            destination_mac_text = str(destination_mac)
            destination_macs.add(destination_mac_text)
            destination_mac_counts[destination_mac_text] += 1
        if registry is not None:
            target_id = registry.target_id_for_mac(source_mac)
            if target_id is not None:
                target_matches[target_id] += 1

    dot11_frame_count = parse_successes
    parse_failure_rate = parse_failures / packets_scanned if packets_scanned else 0.0
    radiotap_coverage = {
        field: _coverage(field_present[field], packets_scanned) for field in RADIOTAP_FIELDS
    }
    summary: dict[str, Any] = {
        "capture_file_count": len(capture_files),
        "capture_files": capture_files,
        "packets_scanned": packets_scanned,
        "parse_successes": parse_successes,
        "parse_failures": parse_failures,
        "parse_failure_rate": parse_failure_rate,
        "dot11_frame_count": dot11_frame_count,
        "frame_family_counts": dict(frame_counts),
        "protected_frame_count": protected_frame_count,
        "channel_frequencies": sorted(channel_frequencies),
        "radiotap_coverage": radiotap_coverage,
        "source_mac_count": len(source_macs),
        "destination_mac_count": len(destination_macs),
        "source_mac_counts": _sorted_counts(source_mac_counts),
        "destination_mac_counts": _sorted_counts(destination_mac_counts),
        "target_match_total": sum(target_matches.values()),
        "target_match_counts": dict(sorted(target_matches.items())),
        "warnings": [],
    }
    summary["warnings"] = inspection_warnings(summary, registry=registry)
    return summary


def inspection_warnings(
    summary: dict[str, Any],
    *,
    registry: TargetRegistry | None = None,
) -> list[str]:
    warnings: list[str] = []
    packets_scanned = int(summary.get("packets_scanned") or 0)
    dot11_frame_count = int(summary.get("dot11_frame_count") or 0)

    if int(summary.get("capture_file_count") or 0) == 0:
        warnings.append("No capture files were found.")
    if packets_scanned and dot11_frame_count == 0:
        warnings.append("No 802.11 Dot11 frames were parsed from the scanned packets.")
    if float(summary.get("parse_failure_rate") or 0.0) >= 0.2:
        warnings.append("At least 20% of scanned packets failed 802.11 parsing.")
    if packets_scanned and _coverage_fraction(summary, "data_rate") == 0.0:
        warnings.append("No Radiotap data-rate values were observed.")
    if packets_scanned and _coverage_fraction(summary, "rssi_or_ssi") == 0.0:
        warnings.append("No Radiotap RSSI/SSI signal values were observed.")
    if dot11_frame_count and int(summary.get("protected_frame_count") or 0) == 0:
        warnings.append(
            "No protected 802.11 frames were observed; encrypted-network runs usually need them."
        )
    if (
        registry is not None
        and dot11_frame_count
        and registry.enabled_target_ids()
        and int(summary.get("target_match_total") or 0) == 0
    ):
        warnings.append("No source MACs matched enabled targets in the configured registry.")
    return warnings


def write_inspection_markdown(path: str | Path, summary: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# peekaboo Capture Inspection",
        "",
        "## Summary",
        "",
        f"- Capture files: `{summary.get('capture_file_count', 0)}`",
        f"- Packets scanned: `{summary.get('packets_scanned', 0)}`",
        f"- Dot11 frames: `{summary.get('dot11_frame_count', 0)}`",
        f"- Parse successes: `{summary.get('parse_successes', 0)}`",
        f"- Parse failures: `{summary.get('parse_failures', 0)}`",
        f"- Protected frames: `{summary.get('protected_frame_count', 0)}`",
        f"- Source MACs observed: `{summary.get('source_mac_count', 0)}`",
        f"- Destination MACs observed: `{summary.get('destination_mac_count', 0)}`",
        f"- Target matches: `{summary.get('target_match_total', 0)}`",
        "",
        "## Radiotap Coverage",
        "",
        "| Field | Present | Missing | Coverage |",
        "| --- | ---: | ---: | ---: |",
    ]
    for field, values in (summary.get("radiotap_coverage") or {}).items():
        lines.append(
            f"| `{field}` | {values.get('present', 0)} | {values.get('missing', 0)} | "
            f"{values.get('coverage', 0.0):.1%} |"
        )

    lines.extend(
        [
            "",
            "## Frame Families",
            "",
            "| Family | Count |",
            "| --- | ---: |",
        ]
    )
    for family, count in (summary.get("frame_family_counts") or {}).items():
        lines.append(f"| `{family}` | {count} |")

    lines.extend(["", "## Warnings", ""])
    warnings = summary.get("warnings") or []
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- None")

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _coverage(present: int, total: int) -> dict[str, float | int]:
    missing = max(total - present, 0)
    return {
        "present": present,
        "missing": missing,
        "coverage": present / total if total else 0.0,
    }


def _coverage_fraction(summary: dict[str, Any], field: str) -> float:
    coverage = (summary.get("radiotap_coverage") or {}).get(field) or {}
    return float(coverage.get("coverage") or 0.0)


def _sorted_counts(counter: Counter) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))
