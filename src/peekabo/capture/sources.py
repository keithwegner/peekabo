"""Offline PCAP/PCAPNG packet sources."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from peekabo.parsing.dot11 import parse_packet_to_record
from peekabo.parsing.radiotap import extract_radiotap_fields
from peekabo.parsing.records import PacketRecord

PCAP_SUFFIXES = {".pcap", ".pcapng", ".cap"}


@dataclass(frozen=True)
class PacketEnvelope:
    source_file: str
    packet_index: int
    packet: Any


def expand_input_paths(paths: Iterable[str | Path]) -> list[Path]:
    expanded: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            expanded.extend(
                sorted(child for child in path.rglob("*") if child.suffix.lower() in PCAP_SUFFIXES)
            )
        elif path.exists():
            expanded.append(path)
    return expanded


def _reader_for_path(path: Path) -> Any:
    try:
        import scapy.layers.dot11  # noqa: F401  # type: ignore
        from scapy.utils import PcapNgReader, PcapReader  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency availability
        raise RuntimeError("Scapy is required for PCAP ingestion") from exc

    if path.suffix.lower() == ".pcapng":
        return PcapNgReader(str(path))
    return PcapReader(str(path))


def iter_packets(paths: Iterable[str | Path]) -> Iterator[PacketEnvelope]:
    for path in expand_input_paths(paths):
        reader = _reader_for_path(path)
        try:
            for index, packet in enumerate(reader):
                yield PacketEnvelope(str(path), index, packet)
        finally:
            reader.close()


def iter_packet_records(paths: Iterable[str | Path]) -> Iterator[PacketRecord]:
    for envelope in iter_packets(paths):
        radiotap = extract_radiotap_fields(envelope.packet)
        yield parse_packet_to_record(
            envelope.packet,
            envelope.source_file,
            envelope.packet_index,
            radiotap,
        )
