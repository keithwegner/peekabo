# peekaboo

`peekaboo` is a passive Python 3.11+ application for per-frame device identification in encrypted 802.11 monitor-mode captures. It classifies each observed 802.11 frame using only unencrypted Radiotap metadata and 802.11 MAC-header fields.

The tool does not decrypt traffic, inspect payloads, inject frames, transmit probes, or exploit networks. Use it only where you are authorized to capture and analyze wireless traffic.

## Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Quick Start

```bash
peekaboo inspect --config configs/example.yaml
peekaboo ingest --config configs/example.yaml
peekaboo features --config configs/example.yaml
peekaboo label --config configs/example.yaml
peekaboo eval-prequential --config configs/example.yaml
peekaboo report --config configs/example.yaml
```

All commands accept a YAML config and targeted path/model overrides. Internal datasets are Parquet by default; CSV export is supported for interoperability.

Run `inspect` before `ingest` on new captures. It checks whether the PCAP inputs contain usable Radiotap and 802.11 metadata, writes `inspect.json` and `inspect.md`, and warns about common first-run problems. If `inspect` or `ingest` reports that no capture files were found, check that the configured `input.paths` exist and contain `.pcap`, `.pcapng`, or `.cap` files.

## Try It With Synthetic Data

The repository includes a generator for a deterministic synthetic Radiotap/802.11 capture. It contains fake MAC addresses and fake frame metadata, so it is safe to use as a first-run demo. The synthetic traffic is intentionally learnable from allowed header-level features such as rate, signal, subtype, and frame size; it is not real-world performance evidence.

```bash
python examples/generate_synthetic_capture.py
peekaboo inspect --config configs/synthetic-demo.yaml
peekaboo ingest --config configs/synthetic-demo.yaml
peekaboo features --config configs/synthetic-demo.yaml
peekaboo label --config configs/synthetic-demo.yaml
peekaboo split --config configs/synthetic-demo.yaml
peekaboo train-online --config configs/synthetic-demo.yaml
peekaboo eval-holdout --config configs/synthetic-demo.yaml
peekaboo classify-file --config configs/synthetic-demo.yaml
peekaboo report --config configs/synthetic-demo.yaml
```

The generated capture is written under `examples/captures/`, and pipeline outputs are written under `runs/`. Both locations are ignored by Git so real captures and generated datasets are not accidentally committed.

After the full demo, `runs/synthetic-demo/` should include:

- `inspect.json` and `inspect.md`: capture preflight diagnostics
- `records.parquet`: normalized Radiotap/Dot11 packet records
- `features.parquet`: paper feature rows with MACs retained for labeling/reporting
- `labeled.parquet`: binary `target` vs. `other` labels
- `train.parquet` and `test.parquet`: chronological holdout split
- `model.pkl`: online model checkpoint
- `metrics.json`: holdout evaluation metrics
- `predictions.parquet`: per-frame predictions
- `rolling.parquet`: rolling target-presence summaries
- `report.md`: Markdown experiment report

## Faithful Feature Policy

The default pipeline preserves MAC addresses for labeling, filtering, reporting, and diagnostics, but drops `source_mac` and `destination_mac` from model features. This prevents trivial identity leakage. To run an explicit ablation or sanity check, set:

```yaml
features:
  leakage_debug: true
```

Reports generated from leakage/debug runs are marked accordingly.

## Model Mapping

The paper used MOA/Weka streaming classifiers. `peekaboo` uses native Python `river` models behind a stable application interface:

| Application ID | Native Python mapping |
| --- | --- |
| `leveraging_bag` | `river.ensemble.LeveragingBaggingClassifier` with a Hoeffding-tree base learner |
| `oza_boost` | `river.ensemble.AdaBoostClassifier`, the closest native online boosting equivalent |
| `oza_boost_adwin` | `river.ensemble.ADWINBoostingClassifier`, falling back to `ADWINBaggingClassifier` if the installed River version lacks ADWIN boosting |
| `adaptive_hoeffding_tree` | `river.tree.HoeffdingAdaptiveTreeClassifier` |

The abstraction leaves room for a later MOA adapter if strict parity is required.

## CLI Commands

```bash
peekaboo ingest
peekaboo inspect
peekaboo features
peekaboo label
peekaboo sample
peekaboo split
peekaboo feature-rank
peekaboo train-online
peekaboo eval-prequential
peekaboo eval-holdout
peekaboo classify-file
peekaboo classify-live
peekaboo report
```

`classify-live` is passive-only. It reads from a preconfigured monitor-mode interface and does not perform channel hopping or interface setup.

## Output Artifacts

Typical runs create:

- capture inspection summaries
- normalized packet records
- feature datasets
- labeled datasets
- model checkpoints
- per-frame predictions
- rolling target-presence summaries
- JSON/CSV metrics
- confusion matrices and plots
- Markdown experiment reports
