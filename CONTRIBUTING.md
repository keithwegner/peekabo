# Contributing

`peekaboo` is a passive-only tool for analyzing unencrypted 802.11 metadata. Contributions should preserve that boundary: no decryption, packet injection, active probing, or payload inspection.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
make install PYTHON=.venv/bin/python
```

Python 3.11 and 3.12 are supported by CI.

## Local Checks

Run the same checks used by CI:

```bash
make check PYTHON=.venv/bin/python
```

Useful individual commands:

```bash
make lint PYTHON=.venv/bin/python
make test PYTHON=.venv/bin/python
make build PYTHON=.venv/bin/python
```

## Pull Requests

Before opening a PR:

- Keep changes scoped and avoid unrelated refactors.
- Do not commit packet captures, generated run outputs, credentials, or local virtual environments.
- Include tests for behavior changes.
- Update README or example config files when CLI behavior changes.
- Describe validation in the PR body.

## Test Data

Prefer synthetic fixtures or sanitized metadata-only examples. Do not publish packet captures that contain private network traffic, device identifiers, or location-sensitive metadata.
