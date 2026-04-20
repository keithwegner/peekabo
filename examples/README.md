# Examples

Generate a deterministic synthetic Radiotap/802.11 capture and run the full demo pipeline:

```bash
python examples/generate_synthetic_capture.py
peekaboo run --config configs/synthetic-demo.yaml
```

The generated capture is written to `examples/captures/synthetic-demo.pcap`, which is ignored by Git. The demo writes its Parquet datasets, model checkpoint, metrics, rolling summaries, live-style replay JSONL streams, `run_manifest.json`, `run_summary.md`, and Markdown report under `runs/synthetic-demo/`, which is also ignored by Git.

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
```

For real authorized monitor-mode captures, place PCAP or PCAPNG files under `examples/captures/`, then point a config at them. The repository intentionally does not include real wireless captures.
