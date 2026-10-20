# Kernels

[← back to the README](../README.md)

Every kernel is a TornadoVM KernelContext kernel launched with an explicit worker grid, except
`gemmHalfTile`, which uses TileContext. The *Hybrid alternative* column says what can replace a kernel
on CUDA.

## Shared

The kernels every model family uses are in [`Kernels.java`](../src/main/java/ai/beestream/jitcast/tornado/Kernels.java).

| Kernel | What it computes | Hybrid alternative |
|---|---|---|
| [`gemm`](../src/main/java/ai/beestream/jitcast/tornado/Kernels.java#L55) | Tiled FP32 GEMM `C (+)= act(A·Wᵀ + b)` with bias / SiLU / accumulate epilogue | cuBLAS `cublasSgemm`, `cublasSgemmTF32` |
| [`gemmHalf`](../src/main/java/ai/beestream/jitcast/tornado/Kernels.java#L109) | Register-blocked FP16-input GEMM, FP32 accumulation, same epilogues and output twin | cuBLAS `cublasGemmExFP16FP32`, `cublasGemmExFP16`; cuBLASLt `ltMatmulFP16`; CUDA Tile `gemmHalfTile` |
| [`gemmBlocked`](../src/main/java/ai/beestream/jitcast/tornado/Kernels.java#L218) | The tiled GEMM register-blocked as `gemmHalf` is, in FP32 throughout (Timestral, from 128 rows) | — |
| [`gemmSimdgroup`](../src/main/java/ai/beestream/jitcast/tornado/Kernels.java#L336) | FP32 GEMM on Apple's simdgroup matrix units (Metal), 32 × 32 tiles staged in threadgroup memory, same epilogues (Timestral, under 64 rows) | — |
| [`gemmSimdgroupWide`](../src/main/java/ai/beestream/jitcast/tornado/Kernels.java#L411) | `gemmSimdgroup` with 64 × 64 tiles whose interior tiles read A and W straight from device memory, optionally over one K slice per work-group (Timestral, from 64 rows) | — |
| [`splitKSum`](../src/main/java/ai/beestream/jitcast/tornado/Kernels.java#L584) | Adds a split-K GEMM's partial tiles and applies the epilogue (Timestral, GEMMs with few tiles) | — |
| [`gemmHalfTile`](../src/main/java/ai/beestream/jitcast/tornado/Kernels.java#L329) | The FP16 GEMM in the CUDA Tile API (`mma` on tensor cores, masked tiles) | is itself the CUDA-only variant |
| [`rmsNorm`](../src/main/java/ai/beestream/jitcast/tornado/Kernels.java#L354) | RMSNorm per row; FP16 twin | — |
| [`swiGlu`](../src/main/java/ai/beestream/jitcast/tornado/Kernels.java#L382) | `SiLU(gate) · up` for the fused up-gate projection; FP16 twin | — |
| [`addInPlace`](../src/main/java/ai/beestream/jitcast/tornado/Kernels.java#L408) | Residual add | — |
| [`biasFill`](../src/main/java/ai/beestream/jitcast/tornado/Kernels.java#L416) | Writes the bias into C before a cuBLAS GEMM with β = 1 | helper for the cuBLAS path |
| [`siluInPlace`](../src/main/java/ai/beestream/jitcast/tornado/Kernels.java#L425) | SiLU after a cuBLAS GEMM; FP16 twin | helper for the cuBLAS path |
| [`toHalf`](../src/main/java/ai/beestream/jitcast/tornado/Kernels.java#L437) | FP32 → FP16 for a GEMM input without a twin | helper for the FP16 GEMMs |
| [`fromHalfEpilogue`](../src/main/java/ai/beestream/jitcast/tornado/Kernels.java#L447) | FP16 C → FP32 with bias / SiLU / accumulate, after an all-FP16 cuBLAS GEMM | helper for `gemmex-h16`, `lt-h16` |
| [`biasSilu`](../src/main/java/ai/beestream/jitcast/tornado/Kernels.java#L472) | Fused bias + SiLU + FP16 twin after a cuBLAS GEMM | helper for the cuBLAS path (`fuse`) |
| [`addRmsNorm`](../src/main/java/ai/beestream/jitcast/tornado/Kernels.java#L496) | Fused residual add + RMSNorm; FP16 twin | — (`fuse`) |

An *FP16 twin* is an FP16 copy of a kernel's output that the kernel writes for the next GEMM. With twins, the FP16 GEMM modes need no separate conversion pass.

## TiRex-2

TiRex-2's own kernels are in [`TiRexKernels.java`](../src/main/java/ai/beestream/jitcast/tirex2/tornado/TiRexKernels.java).

| Kernel | What it computes | Hybrid alternative |
|---|---|---|
| [`scalePatch`](../src/main/java/ai/beestream/jitcast/tirex2/tornado/TiRexKernels.java#L48) | Per-row NaN-aware scaler (nanmean/std, arcsinh, binary rows), patching and the is-present mask; FP16 twin | — |
| [`headLayerNorm`](../src/main/java/ai/beestream/jitcast/tirex2/tornado/TiRexKernels.java#L131) | Per-head LayerNorm: plain, gated by σ(o) (mLSTM output), or added into the residual stream | — |
| [`mlstmCell`](../src/main/java/ai/beestream/jitcast/tirex2/tornado/TiRexKernels.java#L186) | mLSTM cell in the stabilised parallel form, one work-group per time step, with suffix-sum log-decays (up to 512 patches); used when the chunkwise grid would be too small to fill the GPU | — |
| [`mlstmChunk`](../src/main/java/ai/beestream/jitcast/tirex2/tornado/TiRexKernels.java#L277) | The same cell chunkwise, as in flash attention: blocks of 16 query steps share each key/value block staged in local memory, with an online stabiliser and block-relative suffix sums; the default (`-Dtirex.tornado.mlstm=auto`) | — |
| [`slstmRecurrence`](../src/main/java/ai/beestream/jitcast/tirex2/tornado/TiRexKernels.java#L439) | sLSTM recurrence over all steps, one work-group per (row block, head), each R read shared by the rows | — |
| [`slstmRecurrenceHalfR`](../src/main/java/ai/beestream/jitcast/tirex2/tornado/TiRexKernels.java#L516) | The same with FP16 recurrent weights | — |
| [`slstmRecurrenceRegR`](../src/main/java/ai/beestream/jitcast/tirex2/tornado/TiRexKernels.java#L602) | The same with half of each thread's FP16 R values held in registers for the whole sequence (generated scalars), so each step streams only the other half; the default (`slstmR=fp16-reg`) where the streamed kernel would take fewer than 8 rows per group, up to 2 rows per group, on devices with 512-thread work-groups | — |
| [`unpatch`](../src/main/java/ai/beestream/jitcast/tirex2/tornado/TiRexKernels.java#L806) | Patches back to a sequence, then `sinh(clamp(y))·scale + loc` | — |
| [`gatherRows`](../src/main/java/ai/beestream/jitcast/tirex2/tornado/TiRexKernels.java#L831) | Gathers the known-covariate rows, time-reversed for the reverse cell pass; FP16 twin | — |
| [`scatterRows`](../src/main/java/ai/beestream/jitcast/tirex2/tornado/TiRexKernels.java#L848) | Scatters the combined rows back | — |
| [`concatForward`](../src/main/java/ai/beestream/jitcast/tirex2/tornado/TiRexKernels.java#L863) | Forward and reverse outputs, concatenated for the combination layer; FP16 twin | — |
| [`qkNorm`](../src/main/java/ai/beestream/jitcast/tirex2/tornado/TiRexKernels.java#L892) | Weightless RMS normalisation of q and k per head, several vectors per pass | — |
| [`variateAttention`](../src/main/java/ai/beestream/jitcast/tirex2/tornado/TiRexKernels.java#L935) | Flash-style attention across the rows of each group, masked (covariates don't attend to targets), costing O(Σ group²); FP16 twin | — |
| [`headNormAddRmsNorm`](../src/main/java/ai/beestream/jitcast/tirex2/tornado/TiRexKernels.java#L1032) | Fused sLSTM head LayerNorm + residual add + RMSNorm; FP16 twin (needs 512 threads per work-group) | — (`fuse`) |

## Timestral

Timestral's own kernels are in [`TimestralKernels.java`](../src/main/java/ai/beestream/jitcast/timestral/tornado/TimestralKernels.java); its GEMMs, RMSNorm and SwiGLU are the shared ones above.

| Kernel | What it computes |
|---|---|
| [`assembleTokens`](../src/main/java/ai/beestream/jitcast/timestral/tornado/TimestralKernels.java#L53) | Each row's encoder tokens (the patch embedding) followed by its decoder tokens (the forecast query) |
| [`rope`](../src/main/java/ai/beestream/jitcast/timestral/tornado/TimestralKernels.java#L77) | Partial RoPE on q and k in place, from a host-built cos/sin table indexed by position |
| [`timeAttention`](../src/main/java/ai/beestream/jitcast/timestral/tornado/TimestralKernels.java#L114) | Flash-style attention along time per row and head, 16 queries per work-group, with the causal, key-validity and context/horizon masks |
| [`splitHeads`](../src/main/java/ai/beestream/jitcast/timestral/tornado/TimestralKernels.java#L342), [`mergeHeads`](../src/main/java/ai/beestream/jitcast/timestral/tornado/TimestralKernels.java#L443) | Copy q, k, v to heads-major arrays and the outputs back, around the cuBLAS time attention (CUDA, long contexts) |
| [`attentionSoftmax`](../src/main/java/ai/beestream/jitcast/timestral/tornado/TimestralKernels.java#L369) | The masks of `timeAttention` and the softmax, in place on the scores of the cuBLAS time attention |
| [`variateAttention`](../src/main/java/ai/beestream/jitcast/timestral/tornado/TimestralKernels.java#L233) | Flash-style attention across the rows of each series per token slot, with the covariate firewalls |
| [`gate`](../src/main/java/ai/beestream/jitcast/timestral/tornado/TimestralKernels.java#L335) | The attention output's sigmoid gate |
| [`gatherDecoder`](../src/main/java/ai/beestream/jitcast/timestral/tornado/TimestralKernels.java#L343) | The decoder tokens, for the quantile head |
