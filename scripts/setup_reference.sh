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
# convert the checkpoint, dump CPU + CUDA goldens and record NXAI's baseline and cold start.
#
#   setup_reference.sh <workdir>
#
# PYTHON picks the interpreter (default python3.11), TORCH_INDEX the PyTorch wheel index (default CUDA 13.0),
# TIREX_REF the tirex-2 tag or commit (default v0.4.0). Triton builds a C helper, so the interpreter needs its
# development headers (a uv-managed Python has them).
# Re-runnable: each step skips work that is already done. Results land in <workdir>/results; the work
# directory is then the one scripts/eval/acc_setup.sh, eval_nvidia.sh and acc_run.sh take.
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

# 4. NXAI baseline ----------------------------------------------------------------------------
B="$HERE/bench_nxai.py"
[ -f workloads.safetensors ] || python "$B" workloads --out workloads.safetensors
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv | tee results/gpu_before_bench.txt
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv | tee -a results/gpu_before_bench.txt
for cfg in eager compile; do
    log "bench $cfg"
    python "$B" run --workloads workloads.safetensors --device cuda --config "$cfg" \
        --scenarios throughput,multivariate --out "results/nxai-$cfg.json" 2>&1 | grep -v Warning | tee "results/nxai-$cfg.log"
done
log "bench flex (multivariate)"
python "$B" run --workloads workloads.safetensors --device cuda --config flex \
    --scenarios multivariate --out results/nxai-flex.json 2>&1 | grep -v Warning | tee results/nxai-flex.log

# 5. Cold start: warm extension cache x10, cold FlashRNN build cache x3 -----------------------
log "cold start"
rm -f results/nxai-cold-warmcache.jsonl results/nxai-cold-coldcache.jsonl
for i in $(seq 10); do
    BENCH_EXEC_START_NS=$(now_ns) python "$B" coldstart --device cuda --out results/nxai-cold-warmcache.jsonl >/dev/null 2>&1
done
for i in $(seq 3); do
    ext=$(mktemp -d)
    TORCH_EXTENSIONS_DIR="$ext" TRITON_CACHE_DIR="$ext/triton" BENCH_EXEC_START_NS=$(now_ns) \
        python "$B" coldstart --device cuda --out results/nxai-cold-coldcache.jsonl >/dev/null 2>&1
    rm -rf "$ext"
done
python - <<'EOF' | tee results/cold_summary.txt
import json, statistics
for name in ("warmcache", "coldcache"):
    rows = [json.loads(l) for l in open(f"results/nxai-cold-{name}.jsonl")]
    keys = ("import_ms", "load_model_ms", "first_forecast_ms", "exec_to_first_forecast_ms")
    print(name, {k: round(statistics.median(r[k] for r in rows)) for k in keys})
EOF
log "done: $WORK/results"
