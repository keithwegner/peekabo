# peekabo

`peekabo` is a passive Python 3.11+ application for per-frame device identification in encrypted 802.11 monitor-mode captures. It classifies each observed 802.11 frame using only unencrypted Radiotap metadata and 802.11 MAC-header fields.

The tool does not decrypt traffic, inspect payloads, inject frames, transmit probes, or exploit networks. Use it only where you are authorized to capture and analyze wireless traffic.

## Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Quick Start

```bash
peekabo ingest --config configs/example.yaml
peekabo features --config configs/example.yaml
peekabo label --config configs/example.yaml
peekabo eval-prequential --config configs/example.yaml
peekabo report --config configs/example.yaml
```

All commands accept a YAML config and targeted path/model overrides. Internal datasets are Parquet by default; CSV export is supported for interoperability.

If `ingest` reports that no capture files were found, check that the configured `input.paths` exist and contain `.pcap`, `.pcapng`, or `.cap` files.

## Try It With Synthetic Data

The repository includes a generator for a tiny synthetic Radiotap/802.11 capture. It contains fake MAC addresses and fake frame metadata, so it is safe to use as a first-run demo.

```bash
python examples/generate_synthetic_capture.py
peekabo ingest --config configs/synthetic-demo.yaml
peekabo features --config configs/synthetic-demo.yaml
peekabo label --config configs/synthetic-demo.yaml
```

The generated capture is written under `examples/captures/`, and pipeline outputs are written under `runs/`. Both locations are ignored by Git so real captures and generated datasets are not accidentally committed.

## Faithful Feature Policy

The default pipeline preserves MAC addresses for labeling, filtering, reporting, and diagnostics, but drops `source_mac` and `destination_mac` from model features. This prevents trivial identity leakage. To run an explicit ablation or sanity check, set:

```yaml
features:
  leakage_debug: true
```

Reports generated from leakage/debug runs are marked accordingly.

## Model Mapping

The paper used MOA/Weka streaming classifiers. `peekabo` uses native Python `river` models behind a stable application interface:

| Application ID | Native Python mapping |
| --- | --- |
| `leveraging_bag` | `river.ensemble.LeveragingBaggingClassifier` with a Hoeffding-tree base learner |
| `oza_boost` | `river.ensemble.AdaBoostClassifier`, the closest native online boosting equivalent |
| `oza_boost_adwin` | `river.ensemble.ADWINBoostingClassifier`, falling back to `ADWINBaggingClassifier` if the installed River version lacks ADWIN boosting |
| `adaptive_hoeffding_tree` | `river.tree.HoeffdingAdaptiveTreeClassifier` |

The abstraction leaves room for a later MOA adapter if strict parity is required.

## CLI Commands

```bash
peekabo ingest
peekabo features
peekabo label
peekabo sample
peekabo split
peekabo feature-rank
peekabo train-online
peekabo eval-prequential
peekabo eval-holdout
peekabo classify-file
peekabo classify-live
peekabo report
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
