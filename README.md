# wifi-id

`wifi-id` is a passive Python 3.11+ application for per-frame device identification in encrypted 802.11 monitor-mode captures. It follows the method described in *Identification in Encrypted Wireless Networks Using Supervised Learning* by Christopher Swartz and Anupam Joshi: classify each observed 802.11 frame using only unencrypted Radiotap metadata and 802.11 MAC-header fields.

The tool does not decrypt traffic, inspect payloads, inject frames, transmit probes, or exploit networks. Use it only where you are authorized to capture and analyze wireless traffic.

## Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Quick Start

```bash
wifi-id ingest --config configs/example.yaml
wifi-id features --config configs/example.yaml
wifi-id label --config configs/example.yaml
wifi-id eval-prequential --config configs/example.yaml
wifi-id report --config configs/example.yaml
```

All commands accept a YAML config and targeted path/model overrides. Internal datasets are Parquet by default; CSV export is supported for interoperability.

## Faithful Feature Policy

The default pipeline preserves MAC addresses for labeling, filtering, reporting, and diagnostics, but drops `source_mac` and `destination_mac` from model features. This prevents trivial identity leakage. To run an explicit ablation or sanity check, set:

```yaml
features:
  leakage_debug: true
```

Reports generated from leakage/debug runs are marked accordingly.

## Model Mapping

The paper used MOA/Weka streaming classifiers. `wifi-id` uses native Python `river` models behind a stable application interface:

| Application ID | Native Python mapping |
| --- | --- |
| `leveraging_bag` | `river.ensemble.LeveragingBaggingClassifier` with a Hoeffding-tree base learner |
| `oza_boost` | `river.ensemble.AdaBoostClassifier`, the closest native online boosting equivalent |
| `oza_boost_adwin` | `river.ensemble.ADWINBoostingClassifier`, falling back to `ADWINBaggingClassifier` if the installed River version lacks ADWIN boosting |
| `adaptive_hoeffding_tree` | `river.tree.HoeffdingAdaptiveTreeClassifier` |

The abstraction leaves room for a later MOA adapter if strict parity is required.

## CLI Commands

```bash
wifi-id ingest
wifi-id features
wifi-id label
wifi-id sample
wifi-id split
wifi-id feature-rank
wifi-id train-online
wifi-id eval-prequential
wifi-id eval-holdout
wifi-id classify-file
wifi-id classify-live
wifi-id report
```

`classify-live` is passive-only. It reads from a preconfigured monitor-mode interface and does not perform channel hopping or interface setup.

## Output Artifacts

Typical runs create:

- normalized packet records
- feature datasets
- labeled datasets
- model checkpoints
- per-frame predictions
- rolling target-presence summaries
- JSON/CSV metrics
- confusion matrices and plots
- Markdown experiment reports
