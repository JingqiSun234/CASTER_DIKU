# CASTER code and source data

Python 3.10, CUDA, and the three local model checkpoints listed in
`external_assets.example.yaml` are required for the complete run. Model
weights are not included. The runner checks the checkpoint contents before
starting model-dependent stages. Use the CUDA 12.4 build of PyTorch 2.5.1.

Install `requirements-gpu.txt`, copy `external_assets.example.yaml` outside this directory, and set the local checkpoint paths.

List the stages:

```bash
python3 run.py list
```

Inspect the commands without executing them:

```bash
python3 run.py plan --work-dir /path/to/work --external-assets /path/to/assets.yaml
```

Run all stages in a new work directory:

```bash
python3 run.py run --work-dir /path/to/work --external-assets /path/to/assets.yaml
```

Use `--from-stage`, `--to-stage`, and `--resume` for staged execution.

The run covers Benchmark A, B-COVID, and B-FLU. It produces four numeric
summary CSV files and the nine result figures in the selected work directory.

The package contains source inputs and executable code only. Generated files
are written to the selected work directory.
