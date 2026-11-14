#!/usr/bin/env bash
# Copyright (c) 2026 Beestream Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The reference setup on a CUDA machine: install NXAI's TiRex-2 in a venv, check that its CUDA kernels build,
# convert the checkpoint and dump CPU + CUDA goldens.
#
#   setup_reference.sh <workdir>
#
# PYTHON picks the interpreter (default python3.11), TORCH_INDEX the PyTorch wheel index (default CUDA 13.0),
# TIREX_REF the tirex-2 tag or commit (default v0.4.0). Triton builds a C helper, so the interpreter needs its
# development headers (a uv-managed Python has them).
# Re-runnable: each step skips work that is already done. Results land in <workdir>/results. NXAI's
# speed baseline and the benchmark harnesses that take this work directory are in the jitCast-benchmarking repository.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${1:?usage: setup_reference.sh <workdir>}"; mkdir -p "$WORK"; WORK=$(cd "$WORK" && pwd)
PY="${PYTHON:-python3.11}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
TIREX_REF="${TIREX_REF:-v0.4.0}"
mkdir -p "$WORK/results"
cd "$WORK"

log() { echo "[$(date +%H:%M:%S)] $*"; }
now_ns() { python3 -c 'import time; print(time.time_ns())'; }   # date +%s%N is GNU-only

# 1. Environment ------------------------------------------------------------------------
if [ ! -d venv ]; then
    log "creating venv with $PY"
    "$PY" -m venv venv
fi
# shellcheck disable=SC1091
. venv/bin/activate
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
if ! python -c "import tirex2" 2>/dev/null; then
    log "installing torch ($TORCH_INDEX) and tirex-2"
    pip install -q --upgrade pip
    pip install -q torch --index-url "$TORCH_INDEX"
    [ -d tirex-2 ] || git clone -q https://github.com/NX-AI/tirex-2
    git -C tirex-2 checkout -q "$TIREX_REF"
    pip install -q -e ./tirex-2 safetensors pytest
fi

{
    echo "date: $(date -Is)"
    echo "host: $(hostname)"
    nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader
    nvcc --version | tail -1
    python -c "import torch,triton,flashrnn,tirex2;print('torch',torch.__version__,'cuda',torch.version.cuda,'triton',triton.__version__)"
    pip list 2>/dev/null | grep -i -E '^(torch|triton|xlstm|mlstm.kernels|flashrnn|tirex-2|numpy) '
    (cd tirex-2 && echo "tirex-2 commit: $(git rev-parse --short HEAD)")
} | tee results/env.txt

# 2. Do the NXAI CUDA kernels build and run on this GPU? ------------------------------------
# FlashRNN compiles its fused sLSTM kernel with nvcc on first use; mLSTM uses Triton kernels (mlstm_kernels; xlstm before tirex-2 0.4.0).
log "CUDA kernel check (FlashRNN nvcc build + Triton mLSTM)"
python - <<'EOF' 2>&1 | tee results/cuda_kernel_check.txt
import time, torch
from tirex2 import TimeseriesType, load_model
t = time.perf_counter()
fm = load_model("NX-AI/TiRex-2", device="cuda")
ts = TimeseriesType(torch.randn(1, 512).cumsum(-1), None, None)
f = fm.forecast([ts], prediction_length=32, output_type="torch")[0]
torch.cuda.synchronize()
print(f"first CUDA forecast OK in {time.perf_counter() - t:.1f} s, shape {tuple(f.shape)}, finite={bool(torch.isfinite(f).all())}")
s = fm.model.stack[1].time_mixer.fwd_cell
print("sLSTM backend:", s.config.backend, "dtype:", s.config.dtype)
m = fm.model.stack[0].time_mixer.fwd_cell.mlstm_backend.config
print("mLSTM kernels:", m.chunkwise_kernel, m.sequence_kernel, m.step_kernel, m.autocast_kernel_dtype)
EOF

# 3. Weights + goldens ------------------------------------------------------------------------
[ -f tirex2.safetensors ] || python "$HERE/convert_ckpt.py" --out tirex2.safetensors
for dev in cpu cuda; do
    [ -f "golden-$dev/manifest.json" ] || python "$HERE/dump_golden.py" --out "golden-$dev" --device "$dev" \
        --reference-json tirex-2/test/references/reference_input.json
done
log "NXAI's own reference tests on Linux CPU"
(cd tirex-2 && python -m pytest -q test/test_references.py 2>&1 | tail -3) | tee results/nxai_reference_tests.txt
python - <<'EOF' | tee results/golden_cpu_vs_cuda.txt
import json, torch
from safetensors.torch import load_file
man = json.load(open("golden-cpu/manifest.json"))
print(f"{'case':28s} {'max|cpu-cuda|':>14s} {'rel':>10s}")
for case, info in man["cases"].items():
    a, b = load_file(f"golden-cpu/{case}.safetensors"), load_file(f"golden-cuda/{case}.safetensors")
    for i in range(info["num_series"]):
        x, y = a[f"forecast.{i}"], b[f"forecast.{i}"]
        d = (x - y).abs().max().item()
        print(f"{case + '.' + str(i):28s} {d:14.3e} {d / (x.abs().max().item() + 1e-12):10.3e}")
EOF

log "done: $WORK/results"
