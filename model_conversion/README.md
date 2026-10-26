# ExecuTorch model conversion

This directory contains the complete host-side toolset for converting the
PyTorch model into an ExecuTorch program for an Arm Ethos-U target and
generating the corresponding CMSIS AI layer.

The converter reads the target and output information produced by
CMSIS-Toolbox in a `*.cbuild-mlops.yml` file. It quantizes `model/model.py`,
delegates supported operations to Ethos-U, invokes Vela, and generates the
files needed by the CMSIS build.

[CMSIS-Toolbox MLOps integration](https://open-cmsis-pack.github.io/cmsis-toolbox/build-overview/#mlops-integration)
describes how the input file is generated.

[Model-conversion toolset specification](model_conversion-specification.md)
defines every deliverable in this directory and its behavioral contract.

The workspace and solution directories have distinct roles:

- The **workspace directory** contains `model_conversion/` and `.venv/`.
  One workspace environment is shared by all solutions.
- A **solution directory** contains a `.csolution.yml`, its generated
  `*.cbuild-mlops.yml`, `model/model.py`, Vela configuration, and generated AI
  layer. A workspace may contain multiple solution directories.

## Set up the environment

Run the appropriate wrapper from the workspace directory.

Linux and macOS:

```console
./model_conversion/setup_venv.sh
```

Windows:

```console
.\model_conversion\setup_venv.bat
```

The setup creates `.venv` in the workspace directory. It installs
`requirements.txt` normally, then installs `requirements-arm-tosa.txt` with
`--no-deps` to preserve the FlatBuffers version required by ExecuTorch.

Use `--recreate` to rebuild an existing environment. The optional `--uv` and
`--python VERSION` arguments select uv and a supported Python version:

```console
./model_conversion/setup_venv.sh --uv --python 3.12 --recreate
```

## Convert a model

Pass the MLOps file belonging to the selected solution. For a solution in the
workspace root:

```console
cbuild setup cmsis-executorch.csolution.yml --active SSE-320-U85 --packs
python model_conversion/create_ai_layer.py cmsis-executorch.cbuild-mlops.yml
cbuild cmsis-executorch.csolution.yml --active SSE-320-U85
```

`create_ai_layer.py` automatically restarts itself with the `.venv` interpreter
when necessary. It resolves `.venv` from the workspace directory. The
directory containing the supplied `*.cbuild-mlops.yml` is the solution
directory, regardless of the shell's current working directory. It resolves
`model/model.py`, Vela configuration, pack metadata, and `model.clayer`
relative to that solution directory.

For a solution in a workspace subdirectory:

```console
python model_conversion/create_ai_layer.py solutions/demo/demo.cbuild-mlops.yml
```

## Input

The generated YAML must contain a `cbuild-mlops` mapping. The converter uses:

- `npu.type` and optional `npu.macs`;
- `vela.ini` and `vela.options`;
- `model.clayer` and optional `model.name`;
- optional `model.input-shape` and `model.calibration-samples`.

The matching `*.cbuild-pack.yml` file supplies the selected ExecuTorch pack
version.

## Output

The directory selected by `model.clayer` receives:

- the requested `.clayer.yml` with runtime, backend, and operator components;
- `model_pte.c` and `model_pte.h`, which embed the program as a C array; and
- `model.pte` for inspection.

## Tests

Run the unit tests from the repository root:

```console
python -m unittest discover -s model_conversion/test -p "test_*.py" -v
```

The tests mock model export and do not require Torch, Vela, an FVP, or target
hardware.
