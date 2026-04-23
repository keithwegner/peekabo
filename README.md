# peekaboo

[![CI](https://github.com/keithwegner/peekaboo/actions/workflows/ci.yml/badge.svg)](https://github.com/keithwegner/peekaboo/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/keithwegner/peekaboo/branch/main/graph/badge.svg)](https://codecov.io/gh/keithwegner/peekaboo)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/badge/lint-ruff-46a6ff)](https://docs.astral.sh/ruff/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](./pyproject.toml)
[![Passive only](https://img.shields.io/badge/safety-passive--only-brightgreen)](./SECURITY.md)

`peekaboo` is a passive Python 3.11+ application for per-frame device identification in encrypted 802.11 monitor-mode captures. It classifies each observed 802.11 frame using only unencrypted Radiotap metadata and 802.11 MAC-header fields.

The tool does not decrypt traffic, inspect payloads, inject frames, transmit probes, or exploit networks. Use it only where you are authorized to capture and analyze wireless traffic.

## Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Run With Docker

Docker is the quickest way to run the CLI, synthetic demo, and offline/replay workflows without installing Python locally:

```bash
docker compose run --rm -T generate-synthetic
docker compose run --rm -T peekaboo run --config configs/synthetic-demo.yaml
docker compose run --rm -T peekaboo dashboard --config configs/synthetic-demo.yaml
```

The published image is available from GHCR after the Docker workflow has run on `main`:

```bash
docker run --rm ghcr.io/keithwegner/peekaboo:latest --help
```

The Compose file bind-mounts `./configs`, `./examples/captures`, and `./runs` into the container. Put authorized local PCAP/PCAPNG captures under `examples/captures/` or point a mounted config at them; generated datasets, reports, dashboards, and models are written under `runs/`.

Docker v1 is for CLI, offline capture, synthetic demo, replay, report, comparison, calibration, and dashboard workflows. It does not configure adapters, channel hop, decrypt traffic, inspect payloads, inject frames, transmit probes, or provide privileged live monitor-mode capture orchestration.

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

The repository includes a generator for a deterministic synthetic Radiotap/802.11 capture. It contains fake MAC addresses and fake frame metadata, so it is safe to use as a first-run demo. The default 120-frame story includes a phone arriving, quiet background chatter, a phone browsing burst, TV streaming, weak edge-of-house traffic, retries, channel changes, RSSI drift, and multiple fake destinations. The synthetic traffic is intentionally learnable from allowed header-level features such as rate, signal, subtype, and frame size; it is not real-world performance evidence.

```bash
python examples/generate_synthetic_capture.py
peekaboo run --config configs/synthetic-demo.yaml
peekaboo dashboard --config configs/synthetic-demo.yaml
peekaboo calibrate-presence --config configs/synthetic-demo.yaml
peekaboo compare --config configs/synthetic-demo.yaml
peekaboo run --config configs/synthetic-multitarget.yaml
peekaboo calibrate-presence --config configs/synthetic-multitarget.yaml --all-targets
peekaboo presence-replay --config configs/synthetic-multitarget.yaml --all-targets
```

The generated capture is written under `examples/captures/`, and pipeline outputs are written under `runs/`. Both locations are ignored by Git so real captures and generated datasets are not accidentally committed.

`peekaboo compare` runs the configured paper-style model/fraction comparison over the labeled synthetic dataset and writes aggregate results under `runs/synthetic-demo/comparison/`. Synthetic comparison results are useful for smoke testing and onboarding only; they are not real-world performance evidence.

`peekaboo dashboard` writes a static, self-contained, local HTML overview to `runs/synthetic-demo/dashboard/index.html`. It reads existing run artifacts only, requires no web server, and is safe to open locally because it does not perform capture, live monitoring, decryption, payload inspection, probing, injection, adapter setup, or channel hopping.

`peekaboo calibrate-presence` uses labeled predictions, or regenerates them from the labeled dataset and checkpoint when needed, to recommend `windowing` thresholds. It writes `calibration_manifest.json`, `calibration_results.csv`, `calibration_results.json`, `calibration_report.md`, `recommended_windowing.yaml`, and best-effort charts under `runs/synthetic-demo/calibration/`. Copy the recommended `windowing` values into an authorized real-capture config before relying on `presence-live`; synthetic calibration is workflow smoke evidence only.

`configs/synthetic-multitarget.yaml` labels the same synthetic capture as multiclass traffic for `iphone_5_user1`, `lg_tv`, and `other`. Its presence config tracks all enabled registry targets, so replay/live-style output emits separate presence windows for each known target.

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
peekaboo dashboard --config configs/synthetic-demo.yaml
peekaboo calibrate-presence --config configs/synthetic-demo.yaml
peekaboo compare --config configs/synthetic-demo.yaml
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
- `dashboard/index.html`: static local run dashboard
- `calibration/`: presence threshold sweep results, report, recommended windowing YAML, and charts
- `comparison/`: aggregate model/fraction comparison results, report, and trend charts

For multi-target presence, run:

```bash
peekaboo run --config configs/synthetic-multitarget.yaml
peekaboo calibrate-presence --config configs/synthetic-multitarget.yaml --all-targets
peekaboo presence-replay --config configs/synthetic-multitarget.yaml --all-targets
```

Use repeated `--target-class` options to track a subset, for example `--target-class iphone_5_user1 --target-class lg_tv`. `--all-targets` uses enabled target IDs from the target registry and requires a multiclass label mode.

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
peekaboo compare
peekaboo calibrate-presence
peekaboo dashboard
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

Comparison runs:

- `peekaboo compare --config configs/synthetic-demo.yaml`: compare configured model IDs and train fractions over a labeled dataset
- `peekaboo compare --models leveraging_bag --models adaptive_hoeffding_tree --train-fractions 0.1 --train-fractions 0.9`: compare a smaller explicit matrix
- `peekaboo compare --no-prepare`: require an existing labeled dataset instead of preparing missing inputs

Presence calibration:

- `peekaboo calibrate-presence --config configs/synthetic-demo.yaml`: sweep presence thresholds and recommend `windowing` values
- `peekaboo calibrate-presence --all-targets --config configs/synthetic-multitarget.yaml`: calibrate all enabled known targets in a multiclass config
- `peekaboo calibrate-presence --objective mcc --force`: choose thresholds by MCC and overwrite previous calibration artifacts

Static dashboard:

- `peekaboo dashboard --config configs/synthetic-demo.yaml`: generate `dashboard/index.html` for an existing run
- `peekaboo dashboard --force`: overwrite a previously generated dashboard
- The dashboard is a static local file over existing artifacts; it does not start a server or rerun pipeline stages.

`classify-live` is passive-only. It reads from a preconfigured monitor-mode interface and does not perform channel hopping or interface setup.

`presence-live` is also passive-only. It loads a trained checkpoint, reads from an already configured monitor-mode interface, prints concise target-presence state changes, and writes streaming JSONL outputs. Use `presence-replay` on prepared feature rows to test the same runtime behavior without live wireless hardware.

Both `presence-replay` and `presence-live` support `--all-targets` for multiclass checkpoints and repeatable `--target-class` options for selected targets. Without those options, they keep the single-target default.

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
- static HTML dashboards
