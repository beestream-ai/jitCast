# jitllm-tirex

A TornadoVM inference engine for [TiRex-2](https://github.com/NX-AI/tirex-2), NXAI's zero-shot
time-series forecasting model (xLSTM backbone, 9-quantile output), built on jitllm's TornadoVM
kernels and the TornadoVM hybrid CUDA libraries (cuBLAS/cuBLASLt/cuDNN).

Status: **P0 running on cyclone; P1 started**. The reference tooling and the NXAI baseline harness
are in place. The Java module loads the checkpoint and runs the input stage and the patch embedding
on the CPU, matching the reference in every golden case (relative max error ≤ 4e-7).

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
```

## Java layout (`ai.beestream.jitllm.tirex`)

| Package | Contents |
|---|---|
| `api` | `TimeSeries`, `Forecast`, `TiRex2Engine` |
| `io` | `SafeTensors` (memory-mapped reader), `Tensor` |
| `model` | `TiRex2Config` (from the embedded model config), `TiRex2Weights` (binds every tensor by name and shape; unknown or missing tensors fail the load) |
| `cpu` | FP32 CPU reference, the oracle for the device kernels: `Ops`, `InputStage`, `CpuReference` |
| `tornado` | The TornadoVM engine (P1) |

## Scripts (`scripts/`)

| Script | What it does |
|---|---|
| `convert_ckpt.py` | `model.ckpt` + `model-config.yaml` → one `tirex2.safetensors` (FP32, aliases removed, config in the metadata) |
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
   - Stabilised chunkwise mLSTM (chunk 64, padded to 128). Causal, so it is equal to the parallel form over 74 steps; `qk_scale = 64^-0.5`, eps 1e-6.
   - Then a per-head LayerNorm (centred, eps 1e-6, weight only), `h · sigmoid(o)`, and `out_proj` 512→512.
6. **sLSTM cell** (FlashRNN; 4 heads × 128):
   - Gates `i, f, z, o`: each is a block-diagonal headwise 128×128 projection with no bias.
   - Recurrence `R: [4, 128, 4, 128]`, bias `[4, 4, 128]`. States `y, c, n, m` start at zero.
   - Per step:
     ```
     raw = Wx + R·y + b
     logf = logsigmoid(f) + m
     m' = all(n==0) ? i : max(i, logf)
     c' = e^(logf-m')·c + e^(i-m')·tanh(z)
     n' = max(e^(logf-m')·n + e^(i-m'), 1)
     y' = sigmoid(o)·c'/n'
     ```
   - `all(n==0)` is taken over the whole tensor, so it is true only on the first step.
   - Output: per-head LayerNorm (eps 1e-6, weight only).
7. **Variate mixer (AttentionBlock)** runs per token across rows. It is skipped when every group has one row, which is the univariate case.
   - Masks: groups attend only within themselves, and covariate rows cannot attend to target rows.
   - `x += WO·SDPA(rms(WQ·n), rms(WK·n), WV·n)` with `n = RMSNorm(x)` (eps 1e-6). The qk-norm has no weight; the scale is `64^-0.5`.
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
