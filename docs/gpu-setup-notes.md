# GPU setup notes (scratch)

uv.lock resolves the portable CPU torch wheel on purpose. Installing the CUDA
build has to happen after `uv sync`, and any later sync puts the CPU wheel back.

Use `uv run --no-sync ...` afterwards.

Folded into the README.
