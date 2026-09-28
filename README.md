# jitllm-tirex

A TornadoVM inference engine for [TiRex-2](https://github.com/NX-AI/tirex-2), NXAI's zero-shot
time-series forecasting model (xLSTM backbone, 9-quantile output), built on jitllm's TornadoVM
kernels and the TornadoVM hybrid CUDA libraries (cuBLAS/cuBLASLt/cuDNN).

Status: **P0 running on cyclone; P1 in progress**. The reference tooling and the NXAI baseline
harness are in place. `CpuTiRex2Engine.forecast()` runs end to end in Java on the CPU (FP32):
batch assembly, forward pass and postprocessor. It matches the reference in every golden case:
- assembled batch: bit-identical;
- each forward stage: relative error ≤ 5.5e-6;
- final forecasts: relative error ≤ 2.3e-6.

The TornadoVM engine (`tornado.TornadoTiRex2Engine`) runs the full model on the device, including covariates, multi-target series and mixed batches. Its kernels are portable and not yet optimised. It matches the reference in all 13 golden cases on OpenCL and on Metal (Apple M4 Pro):
- model output: relative error ≤ 2.2e-5;
- forecasts: relative error ≤ 8e-6.

Next: the CUDA hybrid path on cyclone (cuBLASLt, CUDA graph, BF16), then the benchmark matrix against NXAI.

| Phase | Scope |
|---|---|
| P0 | Reference install, checkpoint converter, golden activations, NXAI baseline on cyclone |
| P1 | Univariate forward pass in FP32, matching the goldens |
| P2 | Covariates / multivariate: reverse cell on known covariates, variate attention |
| P3 | Performance: CUDA graph, cuBLASLt epilogues, BF16 tensor cores, fused sLSTM step |
| P4 | Benchmark matrix vs NXAI, accuracy parity (GIFT-Eval univariate, fev-bench mini), report |

## Build and test

```bash
# once, from the repository root: install jitllm against the TornadoVM 7.0.1 release
./mvnw -Prelease -DskipTests -Dgpg.skip=true install
# this module; the golden tests are skipped unless both paths are given
cd jitllm-tirex
../mvnw test -Dtirex.weights=/path/tirex2.safetensors -Dtirex.golden=/path/golden-cpu
# device tests: kernel parity and the TornadoVM engine against the goldens (device <driver>:<device>)
TORNADOVM_HOME=/path/sdk ../mvnw test -Paccel-tests -Dtest='*AccelTest' -Dtirex.tornado.device=1:0 \
    -Dtirex.weights=/path/tirex2.safetensors -Dtirex.golden=/path/golden-cpu
```

Two TornadoVM 7.0.1 constraints shape the engine:
- **Graph limits.** A task graph holds at most 256 tasks and an 8 KiB bytecode buffer, so the forward pass is split into 14 graphs that pass device buffers along with `persistOnDevice` / `consumeFromDevice`.
- **A codegen bug.** A short-circuit `a || b` condition inside a loop is emitted without a `continue`, so the variate-attention mask is computed as an int instead.

The variate attention loops only over the rows of the query's own group, which costs O(Σ group²). NXAI's dense masked attention costs O(rows²).

The host uploads values with NaN replaced by 0, plus an is-present mask, so no kernel ever tests for NaN. Device compilers may assume NaNs don't occur (Metal's default fast math does, and so does CUDA's `--use_fast_math`), which makes `v != v` checks unreliable.

## Java layout (`ai.beestream.jitllm.tirex`)

| Package | Contents |
|---|---|
| `api` | `TimeSeries`, `Forecast`, `TiRex2Engine` |
| `io` | `SafeTensors` (memory-mapped reader), `Tensor` |
| `model` | `TiRex2Config` (from the embedded model config), `TiRex2Weights` (binds every tensor by name and shape; unknown or missing tensors fail the load) |
| `cpu` | FP32 CPU reference, the oracle for the device kernels: `Ops`, `InputStage`, `Cells` (mLSTM, sLSTM), `CpuReference` (the full forward pass) |
| `host` | `ForecastPipeline` (`ForecastModel.forecast`: 512-series windows, batch assembly and padding, horizon slicing, sign-flip TTA) and `PostProcessor` (trend detection, differencing, band calibration) — shared by every engine |
| `cpu` … | `CpuTiRex2Engine` wraps `CpuReference` in the pipeline |
| `tornado` | `TiRexKernels` (scale+patch, tiled GEMM with bias/SiLU/accumulate epilogues, RMSNorm, per-head LayerNorm, mLSTM cell, sLSTM gates + recurrence, SwiGLU, un-patch), `PackedWeights` (one device buffer; fused q\|k\|v\|o\|i\|f and up_gate\|up), `TornadoForward` (a plan per batch shape — rows, known covariates, rows in multi-row groups — built as a chain of task graphs: input, one per block, output), `TornadoTiRex2Engine` |

## Scripts (`scripts/`)

| Script | What it does |
|---|---|
| `convert_ckpt.py` | `model.ckpt` + `model-config.yaml` → one `tirex2.safetensors` (FP32, aliases removed, config in the metadata) |
| `spec_forward.py` | An explicit PyTorch spec of `forward()` (basic ops only, no xlstm/flashrnn) checked stage by stage against the goldens; the Java reference is a port of it |
| `dump_golden.py` | Runs the NXAI model on 13 cases and saves the inputs, the batch passed to `forward()`, the activations of every stage, the raw model output and the final forecasts |
| `bench_nxai.py` | NXAI baseline: `workloads` (shared inputs), `run` (latency/throughput, e2e + forward-only), `coldstart` |
| `cyclone_p0.sh` | Runs all of P0 on a CUDA box: venv, CUDA-kernel check, conversion, CPU + CUDA goldens, baseline, cold start |

```bash
./scripts/cyclone_p0.sh ~/tirex2
```

The golden cases cover the three NXAI reference scenarios (univariate, multivariate, multitarget)
plus sign-flip TTA, short/full/overlong contexts, trend differencing, binary series, NaN gaps,
ragged batches, and past + future covariates in mixed batches.

## Model facts (checkpoint `NX-AI/TiRex-2`)

- 82.5M unique FP32 parameters (38.4M on the univariate path + 44.1M for the variate mixers and
  combination layers). The state dict holds 397 entries; 114 are aliases (`rev_cell` is
  `fwd_cell`, and the sLSTM `recurrent_kernel`/`bias` proxies duplicate `_recurrent_kernel_`/`_bias_`).
- The input is padded to 2048 context + 320 future steps = 2368 steps = **74 patches of 32**.
  Every forward pass has L = 74 and D = 512.
- 12 blocks in the recipe `[mLSTM, sLSTM] × 6`. Each block is a time mixer followed by a variate mixer.

## Numerics spec (from the reference code, verified against the goldens in P1)

Shapes: `R` = rows = Σ over series of (targets + covariates), `L = 74`, `D = 512`.

1. **Scaler**, per row over the whole padded row:
   - `loc = nanmean(x)` (NaN → 0), `scale = sqrt(nanmean((x-loc)²))` (NaN → 1), clamped to ≥ 1e-8.
   - Binary rows (all values in {0, 1, NaN}) keep `loc = 0, scale = 1` and skip arcsinh.
   - Every other row gets `arcsinh((x-loc)/scale)`.
2. **Patch**: unfold into `[R, 74, 32]`. The input to the embedding is `cat(nan_to_num(x, 0), !isnan(x))` → `[R, 74, 64]`.
3. **ResidualBlock** (input 64→2048→512, output 512→2048→288): `W_o·SiLU(W_h·x + b_h) + b_o + W_r·x + b_r`.
4. **Time mixer (BiXLSTM)**:
   ```
   xn = RMSNorm(x)
   y  = cell(xn)
   y[known] = combination(cat(y[known], flip(cell(flip(xn[known])))))
   x  = x + y
   x  = x + FFN(RMSNorm(x))
   ```
   - `known` = covariate rows with at least one non-NaN future value. The combination layer is 1024→512 with no bias.
   - RMSNorm here is the xLSTM-large one: eps 1e-6, FP32 reduction, weight only.
   - FFN = `proj_down(SiLU(proj_up_gate(x)) * proj_up(x))`, with hidden size 1408.
5. **mLSTM cell** (4 heads; qk head dim 64, v head dim 128):
   - Projections: `q, k: 512→256`, `v, o: 512→512`, `i, f: 512→4` with bias.
   - `i, f` go through `soft_cap(z) = 15·tanh(z/15)`.
   - xlstm runs a chunkwise kernel (chunk 64, padded to 128). Because it is causal, it equals this stabilised parallel form over the 74 steps, with `F[t] = Σ_{u≤t} logσ(f_u)`:
     ```
     logD[t,s] = F[t] − F[s] + i[s]            (s ≤ t)
     m[t]      = max(F[t], max_s logD[t,s])
     C[t,s]    = (q_t·k_s)/8 · exp(logD[t,s] − m[t])
     h_t       = Σ_s C[t,s]·v_s / (max(|Σ_s C[t,s]|, exp(−m[t])) + 1e-6)
     ```
     The `F[t]` term in `m` is the zero initial state's stabiliser. Leaving it out changes the `exp(−m)` floor of the normaliser.
   - Then a per-head LayerNorm (centred, eps 1e-6, weight only), `h · sigmoid(o)`, and `out_proj` 512→512.
6. **sLSTM cell** (FlashRNN; 4 heads × 128):
   - Gates: four block-diagonal headwise 128×128 projections with no bias.
   - FlashRNN reads the gate slots as `(i, f, z, o)`, but TiRex stacks its projections as `(fgate, igate, zgate, ogate)`. So the checkpoint's `fgate` feeds the **input** gate.
   - Recurrence `R: [4, 128, 4, 128]`, bias `[4, 4, 128]`. States `y, c, n, m` start at zero.
   - Per step:
     ```
     raw = Wx + R·y + b
     logf = logsigmoid(f) + m
     m' = t==0 ? i : max(i, logf)
     c' = e^(logf-m')·c + e^(i-m')·tanh(z)
     n' = max(e^(logf-m')·n + e^(i-m'), 1)
     y' = sigmoid(o)·c'/n'
     ```
   - FlashRNN's test is `all(n == 0)` over the whole tensor. Because `n' ≥ 1`, that is exactly `t == 0`.
   - Output: per-head LayerNorm (eps 1e-6, weight only).
7. **Variate mixer (AttentionBlock)** runs per token across rows (4 heads × **128**). Rows alone in their group pass through unchanged, so the whole block is skipped when every group has one row (the univariate case).
   - Masks: groups attend only within themselves, and covariate rows cannot attend to target rows.
   - `x += WO·SDPA(rms(WQ·n), rms(WK·n), WV·n)` with `n = RMSNorm(x)` (eps 1e-6). The qk-norm is `F.rms_norm` with no weight and eps = float32 machine epsilon (1.19e-7). The scale is `128^-0.5`.
   - Then `x += wo·SiLU(wi·RMSNorm(x))`, with 512→2048→512.
8. **Output**:
   - `stack_out_norm` is a T5-style LayerNorm: RMS, eps 1e-7, weight only.
   - Output ResidualBlock → `[R, 74, 9, 32]` → `[R, 9, 2368]`.
   - Rescale: `sinh(clamp(x, ±20))·scale + loc`; binary rows skip the sinh.
   - Keep `[..., -352:-32]`, then the first H steps.
9. **Postprocessor** (host, FP64 OLS):
   - Trend-detect each target on its unpadded history: `|slope_z|·len ≥ 1.5` and `R² ≥ 0.87` → first-difference the row.
   - Output band calibration:
     - Raw rows: `median + √0.90·t^0.02·(q - median)`.
     - Differenced rows: `cumsum(median) + √1.68·sign·√cumsum(offset²)·t^0.01`, plus the last value.
