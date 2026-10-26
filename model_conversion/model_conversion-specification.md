# ExecuTorch model-conversion toolset specification

## Purpose and directory contract

`model_conversion/` is the complete host-side toolset for converting a
solution's PyTorch model into an ExecuTorch program for Arm Ethos-U and
generating its CMSIS AI layer.

The directory above `model_conversion/` is the **workspace directory**. It
owns the shared `.venv`. The directory containing a selected
`*.cbuild-mlops.yml` is the **solution directory**. It owns the model,
solution-specific configuration, pack metadata, and generated AI layer. A
workspace may contain multiple solution directories.

## Deliverables

### `README.md`

The README is the user entry point. It shall give the setup, conversion, and
test commands; explain the workspace/solution distinction; and link to this
specification for the complete directory contract.

### `create_ai_layer.py`

This is the model converter and AI-layer generator. Invoke it with exactly one
MLOps file:

```console
python model_conversion/create_ai_layer.py <solution-dir>/<solution>.cbuild-mlops.yml
```

It shall locate `.venv` in the workspace directory and restart itself with
that environment's interpreter unless it is already running there. It shall
derive the solution directory from the MLOps file rather than from the current
working directory.

The YAML root shall contain `cbuild-mlops`. The converter uses:

- `npu.type`: required Ethos-U type;
- `npu.macs`: optional MAC count, defaulting to 256;
- `vela.ini`: optional path relative to the solution directory;
- `vela.options`: generated command-line options;
- `model.clayer`: required output path relative to the solution directory;
- `model.name`: optional layer group and description fallback;
- `model.input-shape`: optional `x`- or comma-separated dimensions; and
- `model.calibration-samples`: optional integer sample count.

It shall load `get_model` and `get_calibration_inputs` from the solution's
`model/model.py`. It shall quantize the exported graph, calibrate it with all
returned samples, delegate supported operations to Ethos-U, invoke Vela, and
serialize the resulting ExecuTorch program.

`--accelerator-config`, `--system-config`, and `--memory-mode` shall initialize
`EthosUCompileSpec`. Other Vela options shall be forwarded as extra flags.
`--config`, `--output-format`, and `--output-dir` are owned by the converter and
shall be rejected. Duplicate compile-spec options and unexpected positional
tokens shall also be rejected.

The matching `<solution>.cbuild-pack.yml` shall be read from the solution
directory. Its selected `PyTorch::ExecuTorch` version determines the pack used
to map serialized CPU operators to CMSIS components.

The directory selected by `model.clayer` shall be created as needed and shall
receive:

```text
<model.clayer>
model.pte
model_pte.c
model_pte.h
```

The C array shall be aligned to 16 bytes. The layer shall select ExecuTorch
Runtime, Kernel Utils, Kernel Registration, Backend EthosU, and the operator
components detected in the serialized program. An operator with no matching
pack component shall produce a warning.

### `setup_venv.py`

This is the platform-independent environment setup implementation. It shall
create or repair `<workspace>/.venv`, never a solution-local environment. It
shall support Python 3.10 through 3.13 and reject unsupported versions before
dependency installation.

`--recreate` shall rebuild the environment. `--uv` shall use uv for environment
creation and package installation. `--python VERSION` shall select the uv
Python version and shall require `--uv`.

The setup shall install `requirements.txt` first and then install
`requirements-arm-tosa.txt` with `--no-deps`. It shall finish with an import
test of the ExecuTorch Arm quantizer.

### `setup_venv.sh`

This is the Linux and macOS entry point. It shall locate `setup_venv.py`
relative to itself, forward all arguments, use `${PYTHON:-python3}` for the
standard setup, and use an isolated uv launcher when `--uv` is requested.

### `setup_venv.bat`

This is the Windows entry point. It shall locate `setup_venv.py` relative to
itself, forward all arguments, use `%PYTHON%` when provided and `python`
otherwise, and use an isolated uv launcher when `--uv` is requested.

### `requirements.txt`

This is the primary dependency manifest for model conversion. Versions shall
remain compatible with the ExecuTorch Arm backend and the supported Python
range.

### `requirements-arm-tosa.txt`

This is the separately installed TOSA serializer manifest. It shall contain
the serializer and required FlatBuffers pin used by the ExecuTorch Arm
backend. It is deliberately installed with `--no-deps` so package metadata
cannot replace the dependency versions selected by the primary environment.

### `model_conversion-specification.md`

This document is the normative directory and behavior specification. Changes
to any deliverable in `model_conversion/` shall be reflected here.

### `test/test_create_ai_layer.py`

This is the unit-test suite for the converter. It shall cover workspace `.venv`
selection, solution-relative path handling, MLOps parsing, Vela options, pack
version selection, operator component selection, and generated artifacts.
Model export shall be mocked so the unit tests do not require Torch, Vela, an
FVP, or target hardware.

Run it from the workspace directory:

```console
python -m unittest discover -s model_conversion/test -p "test_*.py" -v
```

## Acceptance criteria

- Invocation from any current directory locates the workspace `.venv` and the
  selected solution's files correctly.
- Multiple solutions in workspace subdirectories share `.venv` while keeping
  their models, configuration, metadata, and output independent.
- Both `--name value` and `--name=value` Vela option forms work.
- Invalid and duplicate options terminate with a clear error.
- The resolved pack version follows the exact selector when available.
- Model parameters are converted to the types expected by `model/model.py`.
- Generated C, header, and layer text are deterministic for identical input.
- The complete unit-test suite passes without target hardware.

