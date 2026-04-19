# Examples

Generate a deterministic synthetic Radiotap/802.11 capture and run the full demo pipeline:

```bash
python examples/generate_synthetic_capture.py
peekabo ingest --config configs/synthetic-demo.yaml
peekabo features --config configs/synthetic-demo.yaml
peekabo label --config configs/synthetic-demo.yaml
peekabo split --config configs/synthetic-demo.yaml
peekabo train-online --config configs/synthetic-demo.yaml
peekabo eval-holdout --config configs/synthetic-demo.yaml
peekabo classify-file --config configs/synthetic-demo.yaml
peekabo report --config configs/synthetic-demo.yaml
```

The generated capture is written to `examples/captures/synthetic-demo.pcap`, which is ignored by Git. The demo writes its Parquet datasets, model checkpoint, metrics, rolling summaries, and Markdown report under `runs/synthetic-demo/`, which is also ignored by Git.

For real authorized monitor-mode captures, place PCAP or PCAPNG files under `examples/captures/`, then point a config at them. The repository intentionally does not include real wireless captures.
