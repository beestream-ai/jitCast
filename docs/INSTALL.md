# Installing jitcast

[← back to the README](../README.md)

## Requirements

- **JDK 22 or newer.** jitcast is tested with JDK 25. It builds with `--release 22` and uses
  TornadoVM's `jdk22plus` SDK family.
- **A TornadoVM SDK**, version 8.0.0 or newer, for the backend you want. jitcast builds against the
  8.0.0 API. TornadoVM 8 compiles kernels with TC, its own compiler, in place of Graal: every model
  passes its golden tests on it, it is up to 1.36× faster than 7.3.1 at small batches, and it runs the
  CUDA Tile GEMM (`cutile`) that 7.3.1 cannot compile
  ([PERFORMANCE.md](PERFORMANCE.md#tornadovm-8-against-731)). jitcast still runs on a 7.3.1 SDK, built
  with `-Dtornadovm.version=7.3.1-jdk22plus`.
- **Python 3.11+ with PyTorch**, only for converting TiRex-2 checkpoints, recording goldens and running
  the reference engines as baselines. Create the venv with that interpreter (`python3.11 -m venv`): a
  plain `python3` can be older. NXAI's Triton kernels compile a C helper, so the interpreter needs its
  development headers (`Python.h`); a distribution Python without them fails at the first CUDA
  forecast, and a uv-managed one (`uv python install 3.12`) has them.

## Install TornadoVM

Prebuilt SDKs for OpenCL, CUDA, Metal, or all of them are at
[tornadovm.org/downloads](https://www.tornadovm.org/downloads) and on SDKMAN!. Pick the `jdk22plus`
family:

```bash
sdk install java 25.0.4-tem                     # a JDK 22+ (the SDK does not include one); sets JAVA_HOME
sdk list tornadovm                              # the available <version>-<jdk>-<backend> combinations
sdk install tornadovm 8.0.0-jdk22plus-metal     # macOS (Apple Silicon); or ...-opencl
sdk install tornadovm 8.0.0-jdk22plus-cuda      # NVIDIA (x86-64)
export TORNADOVM_HOME=<the SDK directory>       # SDKMAN! sets this for you
tornado --devices                               # lists the backends and devices, numbered <backend>:<device>
```

The first `tornado` command also writes `$TORNADOVM_HOME/tornado-argfile`, the JVM flags jitcast is
launched with. `tornado --generate-argfile` regenerates it after switching JDK.

### Building TornadoVM

The `develop` branch has no prebuilt SDK, and Linux on AArch64 (for example NVIDIA Grace or Ampere
hosts) has prebuilt CUDA SDKs only from 8.0.0 on. Build one with `make`, from a release tag or `develop`, inside a
Python venv (the build scripts pip-install their helpers), with `JAVA_HOME` at the JDK you will run
jitcast with:

```bash
git clone --depth 1 -b v8.0.0 https://github.com/beehive-lab/TornadoVM && cd TornadoVM    # or -b develop
python3 -m venv .venv && . .venv/bin/activate && pip install requests tqdm
export JVMCI_SOURCE_JDK=<a JDK 21 home>         # releases up to 7.3.1 only: they bundle a JVMCI module built from JDK 21
make jdk22plus BACKEND=cuda,opencl              # or BACKEND=metal,opencl on macOS
export TORNADOVM_HOME=$(ls -d dist/*/tornadovm-*)
$TORNADOVM_HOME/bin/tornado --devices
```

Build jitcast against the SDK's own artifacts with `-Dtornadovm.version=<its version>` (for
example `8.0.1-jdk22plus-dev` for `develop`); the build installs them in `~/.m2`.

### Device numbering

`tornado --devices` numbers the backends in the order the SDK lists them, for example OpenCL `0:0`
and CUDA `1:0` on an NVIDIA machine with both. The accelerated tests (`-Paccel-tests`) give CUDA
priority, so in the tests CUDA is `0:0`. Pick the device per JVM with `-Djitcast.tornado.device`.

The 8.0.0 SDKs for macOS have one backend each, so in the Metal SDK Metal is `0:0` and the default
with no further setting. An SDK built with both Metal and OpenCL lists OpenCL `0:0` and Metal `1:0`. TornadoVM compiles kernels for its
default backend, and the simdgroup GEMMs of Timestral and Chronos-2 exist only on Metal, so run Metal
as the default with `-Dtornado.metal.priority=100`: Metal is then `0:0`. Without it both still run on
Metal `1:0`, with the portable register-blocked GEMMs. A backend's priority is
`-Dtornado.<backend>.priority`, and the backend with the highest priority is numbered `0`.

### NVIDIA extras

- **cuBLAS 13 on Blackwell** (TornadoVM 7.3.1 and earlier; TornadoVM 8 loads the newest CUDA
  libraries first and needs no shim). TornadoVM's cuBLAS provider loads `libcublas.so.12` first. If a CUDA 12.0
  cuBLAS is installed next to CUDA 13, it silently runs generic, non-tensor-core kernels on sm_120.
  Point `LD_LIBRARY_PATH` at a directory where `lib{cublas,cublasLt,cudart}.so.12` link to the CUDA 13
  libraries; `scripts/eval/cublas13_shim.sh <dir>` builds such a shim, and `scripts/eval/tornado_run.sh`
  uses it when `CUBLAS_SHIM=<dir>` is set.
- **CUDA Tile (`cutile`)** needs TornadoVM 8 (7.3.1 fails to compile its GEMM) and toolkit
  13.3+. A user-space install is enough:
  `pip install 'cuda-tile[tileiras]'` into a venv, then pass
  `-Dtornado.cuda.nvcc=<venv>/lib/python3.*/site-packages/nvidia/cu13/bin/nvcc`. The first run
  compiles each GEMM task with `tileiras`, which takes 2–3 minutes per plan layout; after that the CUDA
  code cache serves it.

## Build jitcast

```bash
git clone https://github.com/beestream-ai/jitcast && cd jitcast
./mvnw -q package -DskipTests                                         # TornadoVM 8.0.0 API from Maven Central
./mvnw -q package -DskipTests -Dtornadovm.version=8.0.1-jdk22plus-dev # or the version of an SDK you built
```

## Get the weights

Each model page says where its checkpoints come from:

| Model | Checkpoints | Preparation |
|---|---|---|
| [TiRex-2](models/TIREX2.md#quick-start) | `NX-AI/TiRex-2` and variants on Hugging Face | converted once to one safetensors file (`scripts/convert_ckpt.py`) |
| [Timestral](models/TIMESTRAL.md#quick-start) | `mistralai/Timestral-{5M,28M,123M,209M}` on Hugging Face | none: `config.json` + `model.safetensors` as published |
| [Chronos-2](models/CHRONOS2.md#quick-start) | `amazon/chronos-2`, `autogluon/chronos-2-small` on Hugging Face | none: `config.json` + `model.safetensors` as published |

Next: [running jitcast](USAGE.md).
