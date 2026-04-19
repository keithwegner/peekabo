"""Target registry loading and MAC matching."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from peekaboo.parsing.dot11 import normalize_mac


@dataclass(frozen=True)
class Target:
    target_id: str
    label: str | None
    mac_addresses: tuple[str, ...]
    enabled: bool = True


class TargetRegistry:
    def __init__(self, targets: list[Target]) -> None:
        self.targets = targets
        self._mac_to_target: dict[str, Target] = {}
        for target in targets:
            if not target.enabled:
                continue
            for mac in target.mac_addresses:
                normalized = normalize_mac(mac)
                if normalized is not None:
                    self._mac_to_target[normalized] = target

    @classmethod
    def from_file(cls, path: str | Path) -> TargetRegistry:
        registry_path = Path(path)
        with registry_path.open("r", encoding="utf-8") as handle:
            if registry_path.suffix.lower() == ".json":
                data = json.load(handle)
            else:
                data = yaml.safe_load(handle) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TargetRegistry:
        targets = []
        for raw in data.get("targets", []):
            targets.append(
                Target(
                    target_id=str(raw["target_id"]),
                    label=raw.get("label"),
                    mac_addresses=tuple(
                        normalize_mac(mac) or "" for mac in raw.get("mac_addresses", [])
                    ),
                    enabled=bool(raw.get("enabled", True)),
                )
            )
        return cls(targets)

    def target_for_mac(self, mac: str | None) -> Target | None:
        normalized = normalize_mac(mac)
        if normalized is None:
            return None
        return self._mac_to_target.get(normalized)

    def target_id_for_mac(self, mac: str | None) -> str | None:
        target = self.target_for_mac(mac)
        return None if target is None else target.target_id

    def enabled_target_ids(self) -> set[str]:
        return {target.target_id for target in self.targets if target.enabled}
