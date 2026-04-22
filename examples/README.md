# Examples

Generate a richer deterministic synthetic Radiotap/802.11 capture and run the full demo pipeline:

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

The generated capture is written to `examples/captures/synthetic-demo.pcap`, which is ignored by Git. Its default 120-frame fake traffic story includes a phone arriving, quiet background chatter, a phone browsing burst, TV streaming, weak edge-of-house traffic, retries, channel changes, RSSI drift, and multiple fake destinations. The demo writes its Parquet datasets, model checkpoint, metrics, rolling summaries, live-style replay JSONL streams, `run_manifest.json`, `run_summary.md`, Markdown report, static HTML dashboard, calibration outputs, and aggregate comparison outputs under `runs/synthetic-demo/`, which is also ignored by Git. Synthetic calibration and comparison results are onboarding smoke evidence only, not real-world wireless performance evidence.

The multi-target demo writes separate multiclass outputs under `runs/synthetic-multitarget/` and emits presence rows for each enabled known target in `configs/targets.yaml`.

To run the same workflow manually one command at a time:

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

For real authorized monitor-mode captures, place PCAP or PCAPNG files under `examples/captures/`, then point a config at them. Run `peekaboo dashboard` after a completed run to open a static local artifact overview without starting a server. Run `peekaboo calibrate-presence` on labeled local replay output before relying on `presence-live`; both commands use metadata-only artifacts and do not decrypt, inspect payloads, inject frames, probe networks, configure adapters, or channel hop. The repository intentionally does not include real wireless captures.
