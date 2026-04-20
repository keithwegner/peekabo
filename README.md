# peekaboo

`peekaboo` is a passive Python 3.11+ application for per-frame device identification in encrypted 802.11 monitor-mode captures. It classifies each observed 802.11 frame using only unencrypted Radiotap metadata and 802.11 MAC-header fields.

The tool does not decrypt traffic, inspect payloads, inject frames, transmit probes, or exploit networks. Use it only where you are authorized to capture and analyze wireless traffic.

## Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Local Checks

```bash
make check PYTHON=.venv/bin/python
```

The test command reports coverage for `src/peekaboo` and enforces the configured coverage gate. Use `make coverage PYTHON=.venv/bin/python` for an HTML report under `htmlcov/`.

## Quick Start

```bash
peekaboo run --config configs/example.yaml
```

`peekaboo run` executes the configured pipeline, writes `run_manifest.json` and `run_summary.md`, and records each stage as completed, skipped, or failed. Use `--dry-run` to print the planned stages, `--force` to overwrite existing stage outputs, or `--skip-existing` to reuse completed artifacts.

All commands accept a YAML config and targeted path/model overrides. Internal datasets are Parquet by default; CSV export is supported for interoperability.

Run `inspect` before `ingest` on new captures. It checks whether the PCAP inputs contain usable Radiotap and 802.11 metadata, writes `inspect.json` and `inspect.md`, and warns about common first-run problems. If `inspect` or `ingest` reports that no capture files were found, check that the configured `input.paths` exist and contain `.pcap`, `.pcapng`, or `.cap` files.

## Use With Your Own Capture

Only use local captures from networks and devices you are authorized to monitor. Put PCAP/PCAPNG files under an ignored location such as `examples/captures/local/`; the repository ignores captures and generated run outputs by default.

To bootstrap a runnable config and target registry from an authorized capture:

```bash
peekaboo setup \
  --input examples/captures/local/home.pcapng \
  --target-id my_phone \
  --target-mac aa:bb:cc:dd:ee:ff \
  --label phone \
  --output-dir runs/home \
  --config-output configs/home.yaml \
  --targets-output configs/home-targets.yaml \
  --run
```

`setup` writes `setup_inspect.json` and `setup_candidates.md` under the chosen output directory. If you omit `--target-mac`, it writes the candidate report only, so you can pick a source MAC before generating a target registry. The command does not configure wireless adapters, channel hop, decrypt traffic, inspect payloads, inject frames, or probe networks.

## Try It With Synthetic Data

The repository includes a generator for a deterministic synthetic Radiotap/802.11 capture. It contains fake MAC addresses and fake frame metadata, so it is safe to use as a first-run demo. The synthetic traffic is intentionally learnable from allowed header-level features such as rate, signal, subtype, and frame size; it is not real-world performance evidence.

```bash
python examples/generate_synthetic_capture.py
peekaboo run --config configs/synthetic-demo.yaml
```

The generated capture is written under `examples/captures/`, and pipeline outputs are written under `runs/`. Both locations are ignored by Git so real captures and generated datasets are not accidentally committed.

To run the synthetic demo one stage at a time instead, use:

```bash
peekaboo inspect --config configs/synthetic-demo.yaml
peekaboo ingest --config configs/synthetic-demo.yaml
peekaboo features --config configs/synthetic-demo.yaml
peekaboo label --config configs/synthetic-demo.yaml
peekaboo split --config configs/synthetic-demo.yaml
peekaboo train-online --config configs/synthetic-demo.yaml
peekaboo eval-holdout --config configs/synthetic-demo.yaml
peekaboo classify-file --config configs/synthetic-demo.yaml
peekaboo presence-replay --config configs/synthetic-demo.yaml
peekaboo report --config configs/synthetic-demo.yaml
```

After the full demo, `runs/synthetic-demo/` should include:

- `run_manifest.json` and `run_summary.md`: reproducible run provenance and stage summary
- `inspect.json` and `inspect.md`: capture preflight diagnostics
- `records.parquet`: normalized Radiotap/Dot11 packet records
- `features.parquet`: paper feature rows with MACs retained for labeling/reporting
- `labeled.parquet`: binary `target` vs. `other` labels
- `train.parquet` and `test.parquet`: chronological holdout split
- `model.pkl`: online model checkpoint
- `metrics.json`: holdout evaluation metrics
- `predictions.parquet`: per-frame predictions
- `rolling.parquet`: rolling target-presence summaries
- `replay_predictions.jsonl` and `replay_presence.jsonl`: live-style replay output
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
peekaboo run
peekaboo setup
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
peekaboo presence-replay
peekaboo presence-live
peekaboo report
```

Runner profiles:

- `peekaboo run --profile full`: inspect, ingest, feature extraction, labeling, split, train, holdout evaluation, file classification, replay presence, report
- `peekaboo run --profile prepare`: inspect through split
- `peekaboo run --profile train-eval`: train, holdout evaluation, file classification, report
- `peekaboo run --profile presence-replay`: train and replay presence output

`classify-live` is passive-only. It reads from a preconfigured monitor-mode interface and does not perform channel hopping or interface setup.

`presence-live` is also passive-only. It loads a trained checkpoint, reads from an already configured monitor-mode interface, prints concise target-presence state changes, and writes streaming JSONL outputs. Use `presence-replay` on prepared feature rows to test the same runtime behavior without live wireless hardware.

## Output Artifacts

Typical runs create:

- capture inspection summaries
- normalized packet records
- feature datasets
- labeled datasets
- model checkpoints
- per-frame predictions
- rolling target-presence summaries
- live/replay prediction and presence JSONL streams
- JSON/CSV metrics
- confusion matrices and plots
- Markdown experiment reports
