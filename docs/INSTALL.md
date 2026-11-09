# Installing jitcast

[← back to the README](../README.md)

## Requirements

- **JDK 22 or newer** (tested with JDK 25). jitcast builds with `--release 22` against TornadoVM's
  `jdk22plus` SDKs.
- **TornadoVM 8.1.0**, the version jitcast builds against, for your backend.
- **Python 3.11+ with PyTorch**, only to convert TiRex-2 checkpoints and record test goldens. Create
  the venv with that interpreter (`python3.11 -m venv`).

## Install TornadoVM with SDKMAN!

[SDKMAN!](https://sdkman.io) installs prebuilt TornadoVM SDKs and a JDK, and keeps several versions
side by side:

```bash
curl -s "https://get.sdkman.io" | bash          # once, then open a new shell
sdk install java 25.0.4-tem                     # any JDK 22+; the TornadoVM SDK does not include one
sdk install tornadovm 8.1.0-jdk22plus-cuda      # the SDK for your platform, from the table below
tornado --devices                               # backends and devices, as <backend>:<device>
```

| Platform | SDKMAN! identifier |
|---|---|
| macOS, Apple Silicon | `8.1.0-jdk22plus-metal` (recommended) or `8.1.0-jdk22plus-opencl` |
| Linux x86-64, NVIDIA GPU | `8.1.0-jdk22plus-cuda` |
| Linux x86-64, other OpenCL GPU | `8.1.0-jdk22plus-opencl` |
| Linux x86-64, every backend in one SDK | `8.1.0-jdk22plus-full` |
| Linux AArch64 (such as NVIDIA Grace), Windows | not on SDKMAN!: see below |

Use the `jdk22plus` SDKs; the `jdk21` ones do not run jitcast. SDKMAN! sets `TORNADOVM_HOME` and
`PATH` for the default version, which `sdk default tornadovm <identifier>` changes; `sdk use
tornadovm <identifier>` switches the current shell only, and `sdk list tornadovm` shows what is
available. The first `tornado` command writes `$TORNADOVM_HOME/tornado-argfile`, the JVM flags
jitcast runs with; `tornado --generate-argfile` rewrites it after a JDK change.

### Without SDKMAN!

Every SDK, including CUDA for Linux AArch64 and Windows, is attached to the
[TornadoVM 8.1.0 release](https://github.com/beehive-lab/TornadoVM/releases/tag/v8.1.0) and on
[tornadovm.org/downloads](https://www.tornadovm.org/downloads). Unpack it and point `TORNADOVM_HOME`
at it:

```bash
curl -fLO https://github.com/beehive-lab/TornadoVM/releases/download/v8.1.0/tornadovm-8.1.0-jdk22plus-cuda-linux-aarch64.tar.gz
tar xzf tornadovm-8.1.0-jdk22plus-cuda-linux-aarch64.tar.gz
export TORNADOVM_HOME=$PWD/tornadovm-8.1.0-cuda PATH=$PWD/tornadovm-8.1.0-cuda/bin:$PATH
tornado --devices
```

### Building TornadoVM

Build an SDK from source for TornadoVM's `develop` branch, inside a Python venv, with
`JAVA_HOME` at the JDK you will run jitcast with:

```bash
git clone --depth 1 -b develop https://github.com/beehive-lab/TornadoVM && cd TornadoVM
python3 -m venv .venv && . .venv/bin/activate && pip install requests tqdm
make jdk22plus BACKEND=cuda,opencl              # or BACKEND=metal,opencl on macOS
export TORNADOVM_HOME=$(ls -d dist/*/tornadovm-*)
$TORNADOVM_HOME/bin/tornado --devices
```

Build jitcast against it with `-Dtornadovm.version=<its version>` (for example
`8.1.1-jdk22plus-dev`); the SDK build installs its artifacts in `~/.m2`.

### Device numbering

`tornado --devices` numbers backends in SDK order, for example OpenCL `0:0` and CUDA `1:0` on an
NVIDIA machine with both. Choose per JVM with `-Djitcast.tornado.device`. The accelerated tests
(`-Paccel-tests`) give CUDA priority, so there CUDA is `0:0`.

The prebuilt macOS SDKs have one backend each. An SDK with both Metal and OpenCL lists OpenCL `0:0`
and Metal `1:0`. TornadoVM compiles matrix-unit kernels only for its default backend, so add
`-Dtornado.metal.priority=100` to make Metal `0:0` and enable the simdgroup GEMMs. In general the
backend with the highest `-Dtornado.<backend>.priority` is numbered `0`.

### NVIDIA extras

- **CUDA Tile (`cutile`)** needs toolkit 13.3+. A user-space install is enough:
  `pip install 'cuda-tile[tileiras]'` into a venv, then pass
  `-Dtornado.cuda.nvcc=<venv>/lib/python3.*/site-packages/nvidia/cu13/bin/nvcc`. The first run of
  each plan compiles its GEMMs with `tileiras`, which takes minutes; the CUDA code cache serves later
  runs.
- **cuBLAS 13 on Blackwell, TornadoVM 7.3.1 and earlier.** These load `libcublas.so.12` first; a CUDA
  12.0 cuBLAS installed next to CUDA 13 then runs generic kernels on sm_120. Point `LD_LIBRARY_PATH`
  at a directory where `lib{cublas,cublasLt,cudart}.so.12` link to the CUDA 13 libraries
  (jitCast-benchmarking's
  [`cublas13_shim.sh`](https://github.com/beestream-ai/jitCast-benchmarking/blob/main/scripts/eval/cublas13_shim.sh)
  builds one). TornadoVM 8 needs no shim.

## Build jitcast

```bash
git clone https://github.com/beestream-ai/jitcast && cd jitcast
./mvnw -q package -DskipTests                                         # TornadoVM 8.1.0 API from Maven Central
./mvnw -q package -DskipTests -Dtornadovm.version=8.1.1-jdk22plus-dev # or a develop SDK you built
```

## Get the weights

| Model | Checkpoints | Preparation |
|---|---|---|
| [TiRex-2](models/TIREX2.md#quick-start) | `NX-AI/TiRex-2` and variants | converted once to safetensors (`scripts/convert_ckpt.py`) |
| [Timestral](models/TIMESTRAL.md#quick-start) | `mistralai/Timestral-{5M,28M,123M,209M}` | none: `config.json` + `model.safetensors` as published |
| [Chronos-2](models/CHRONOS2.md#quick-start) | `amazon/chronos-2`, `autogluon/chronos-2-small` | none: `config.json` + `model.safetensors` as published |

Next: [running jitcast](USAGE.md).
