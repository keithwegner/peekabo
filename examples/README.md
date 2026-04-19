# Examples

Generate a tiny synthetic Radiotap/802.11 capture:

```bash
python examples/generate_synthetic_capture.py
peekabo ingest --config configs/synthetic-demo.yaml
peekabo features --config configs/synthetic-demo.yaml
peekabo label --config configs/synthetic-demo.yaml
```

The generated capture is written to `examples/captures/synthetic-demo.pcap`, which is ignored by Git.

For real authorized monitor-mode captures, place PCAP or PCAPNG files under `examples/captures/`, then point a config at them. The repository intentionally does not include real wireless captures.
