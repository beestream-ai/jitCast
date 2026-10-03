# Installing jitrex

[← back to the README](../README.md)

## Requirements
- **JDK 22 or newer.** It is tested with JDK 25. It builds with `--release 22` and uses TornadoVM's `jdk22plus` SDK family.
- **A TornadoVM SDK**, version 7.0.1 or newer, for the backend you want.
- **Python 3.11+ with PyTorch**, once, to convert NXAI's checkpoints. It is also needed for the NXAI baselines and the accuracy benchmarks.

## Install TornadoVM

Prebuilt SDKs for OpenCL, CUDA, Metal, or all of them are at [tornadovm.org/downloads](https://www.tornadovm.org/downloads) and on SDKMAN!. Pick the `jdk22plus` family:

```bash
sdk list tornadovm                              # the available <version>-<jdk>-<backend> combinations
sdk install tornadovm 7.0.1-jdk22plus-metal     # macOS (Apple Silicon); or ...-opencl
sdk install tornadovm 7.0.1-jdk22plus-cuda      # NVIDIA
export TORNADOVM_HOME=<the SDK directory>       # SDKMAN! sets this for you
tornado --devices                               # lists the backends and devices, numbered <backend>:<device>
```

`tornado --generate-argfile` regenerates `$TORNADOVM_HOME/tornado-argfile` (the JVM flags jitrex is launched with) after switching JDK. To build TornadoVM from source, for example for its `develop` branch, see the [TornadoVM installation guide](https://tornadovm.readthedocs.io/en/latest/installation.html).

### NVIDIA extras
- **cuBLAS 13 on Blackwell.** TornadoVM's cuBLAS provider loads `libcublas.so.12` first. If a CUDA 12.0 cuBLAS is installed next to CUDA 13, it silently runs generic, non-tensor-core kernels on sm_120. Point `LD_LIBRARY_PATH` at a directory where `lib{cublas,cublasLt,cudart}.so.12` link to the CUDA 13 libraries; `scripts/eval/cublas13_shim.sh <dir>` builds such a shim, and `scripts/eval/tornado_run.sh` uses it when `CUBLAS_SHIM=<dir>` is set.
- **CUDA Tile (`cutile`)** needs toolkit 13.3+. A user-space install is enough: `pip install 'cuda-tile[tileiras]'` into a venv, then pass `-Dtornado.cuda.nvcc=<venv>/lib/python3.*/site-packages/nvidia/cu13/bin/nvcc`. The first run compiles each GEMM task with `tileiras`, which takes 2–3 minutes per plan layout; after that the CUDA code cache serves it.

- **NXAI's reference engine on a newer CUDA toolkit than the driver.** NXAI's TiRex-2 runs its sLSTM layers through FlashRNN, which compiles its CUDA kernels on first use with the toolkit `CUDA_HOME` points to, and only as PTX (`-gencode arch=compute_80,code=compute_80` in `flashrnn/cuda_init.py`). The driver then compiles that PTX at load time, and a driver cannot compile PTX from a newer toolkit than its own: with CUDA 13.4 on a driver that supports CUDA 13.3, every NXAI forecast fails with `RuntimeError: Errors during CUDA kernel calls forward` (the CUDA error underneath is 222, "the provided PTX was compiled with an unsupported toolchain"). Point FlashRNN at a toolkit no newer than the driver, for example the one PyTorch was built for, and clear its build cache:
  ```bash
  export CUDA_HOME=/usr/local/cuda-13.0 CUDA_PATH=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH
  rm -rf ~/.cache/torch_extensions/py*_cu*
  ```
  This only concerns NXAI's engine, not jitrex. The fix belongs in FlashRNN: compile native code for the GPU next to the PTX (`code=[sm_XY,compute_XY]` from `torch.cuda.get_device_capability()`, or `TORCH_CUDA_ARCH_LIST`), which a newer 13.x toolkit can build for an older 13.x driver, and report the CUDA error string instead of the generic message. Upgrading the driver also avoids it.

The results in these documents were measured on TornadoVM `develop` at 85eb7bed2.

## Build jitrex

```bash
git clone https://github.com/beestream-ai/jitcast && cd jitrex
./mvnw -q package -DskipTests                                     # TornadoVM 7.0.1 API from Maven Central
./mvnw -q package -DskipTests -Dtornadovm.version=7.0.2-jdk22plus-dev   # or another installed version
```

## Get the weights

jitrex reads one safetensors file per checkpoint. `convert_ckpt.py` downloads a checkpoint from Hugging Face and writes that file, with aliases removed and the model config embedded:

```bash
python3 -m venv venv && . venv/bin/activate
pip install torch safetensors huggingface_hub pyyaml   # what convert_ckpt.py needs
python scripts/convert_ckpt.py --ckpt NX-AI/TiRex-2 --out tirex2.safetensors
python scripts/convert_ckpt.py --ckpt NX-AI/TiRex-2-gifteval-zs --out TiRex-2-gifteval-zs.safetensors
```

Next: [running jitrex](USAGE.md).
