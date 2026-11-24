#!/usr/bin/env python3
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

"""Dump golden inputs, per-stage activations and forecasts of Google's TimesFM 2.5 reference.

For each case this writes ``<out>/<case>.safetensors`` holding:
  in.context                        the forecast() inputs, (B, T), NaN-padded on the left for a ragged case
  point / quantiles                 forecast()'s outputs: (B, H) and (B, H, 10) (index 0 the mean head,
                                    1..9 the levels 0.1..0.9)
  call<k>.in.inputs / .in.masks     the k-th TimesFM_2p5_200M_torch_module.decode call's arguments,
                                    (B, C) after the batch assembly (and normalize_inputs), masks as 0/1
  call<k>.pf                        its prefill output, renormalized, (B, patches, 128, 10)
  call<k>.spread                    the last patch's quantile spread, renormalized, (B, 1024, 10)
  call<k>.ar                        the autoregressive outputs, (B, steps, 128, 10) (absent without steps)
  call<k>.normed                    the prefill's normalized patches, (B, patches, 32)
  call<k>.mu / .sigma               the running statistics of each patch, (B, patches)
  call<k>.tokens                    the tokenizer's output, (B, patches, D)
  call<k>.layer<i>                  each transformer layer's output, (B, patches, D)
  call<k>.head_point                the point projection, normalized, (B, patches, 1280)
  call<k>.head_spread               the quantile projection of the last patch, normalized, (B, 1, 10240)
  call<k>.fp32err.<stage>           how far that stage of the float32 reference is from the same call run
                                    in float64 (a scalar, the relative max error over the unmasked
                                    patches): the noise floor a float32 engine is measured against
and ``<out>/manifest.json`` describing every case (H, ForecastConfig, decode horizon per call).

The Java engine reads these files in its unit tests, so the inputs are identical.

Usage: dump_golden.py --src <timesfm checkout>/src --ckpt <dir with model.safetensors> --out golden/
"""

import argparse
import copy
import dataclasses
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


def series(length, seed, trend=0.0, season=24, level=10.0):
    g = np.random.default_rng(seed)
    t = np.arange(length, dtype=np.float64)
    walk = np.cumsum(g.standard_normal(length)) * 0.3
    return (level + walk + 2.0 * np.sin(2 * math.pi * t / season) + trend * t).astype(np.float32)


# The configuration TimesFM's documentation recommends; cases override single fields.
RECOMMENDED = dict(max_context=1024, max_horizon=256, normalize_inputs=True, use_continuous_quantile_head=True,
                   force_flip_invariance=True, infer_is_positive=True, fix_quantile_crossing=True)


def cases():
    """name -> (H, [series], ForecastConfig overrides)."""
    out = {}
    out["univariate_short"] = (16, [series(100, 1)], {})
    out["univariate_long"] = (512, [series(1000, 2, trend=0.01)], {"max_horizon": 512})
    out["multi_step"] = (1300, [series(900, 3)], {"max_horizon": 1408, "use_continuous_quantile_head": False})
    nans = series(500, 5)
    nans[:70] = np.nan  # a late start
    nans[200:260] = np.nan  # a gap
    out["with_nans"] = (40, [nans], {})
    out["ragged_batch"] = (48, [series(n, 10 + i) for i, n in enumerate((50, 333, 1024))], {})
    out["constant"] = (32, [np.full(200, 5.0, dtype=np.float32)], {})
    out["negative"] = (64, [series(400, 6, level=-20.0)], {})
    out["near_zero"] = (64, [np.abs(series(300, 7, level=0.0)) * 0.01], {})
    out["long_context"] = (128, [series(5000, 8)], {"max_context": 4096})
    out["plain"] = (96, [series(700, 9)], dict(normalize_inputs=False, use_continuous_quantile_head=False,
                                              force_flip_invariance=False, infer_is_positive=False,
                                              fix_quantile_crossing=False))
    out["no_flip"] = (64, [series(600, 11)], {"force_flip_invariance": False})
    out["large_scale"] = (64, [series(600, 12, level=1e6) * 1e3], {})
    out["many_series"] = (16, [series(64 + (i % 7) * 13, 100 + i) for i in range(40)], {})
    return out


class Recorder:
    """Records every decode call: its arguments and outputs, and the stages of its prefill forward pass."""

    def __init__(self, module):
        self.module = module
        self.calls = []
        self._current = None
        self._prefill = False
        decode = module.decode

        def wrapped(horizon, inputs, masks):
            self._current = {"horizon": horizon, "in.inputs": inputs, "in.masks": masks.float()}
            self._prefill = True
            pf, spread, ar = decode(horizon, inputs, masks)
            self._current.update({"pf": pf, "spread": spread})
            if ar is not None:
                self._current["ar"] = ar
            self.calls.append(self._current)
            self._current = None
            return pf, spread, ar

        module.decode = wrapped
        module.tokenizer.register_forward_pre_hook(self._tokenizer_in)
        module.tokenizer.register_forward_hook(self._store("tokens"))
        for i, layer in enumerate(module.stacked_xf):
            layer.register_forward_hook(self._store(f"layer{i}", lambda o: o[0]))
        module.output_projection_point.register_forward_hook(self._store("head_point"))
        module.output_projection_quantiles.register_forward_hook(self._store("head_spread", lambda o: o[:, -1:], last=True))

    def _tokenizer_in(self, module, args):
        if self._prefill and self._current is not None:
            x = args[0]
            p = x.shape[-1] // 2
            self._current["normed"] = x[..., :p]

    def _store(self, name, pick=lambda o: o, last=False):
        def hook(module, args, output):
            if self._prefill and self._current is not None:
                self._current[name] = pick(output)
                if last:
                    self._prefill = False
        return hook


def running_stats(inputs, masks, p):
    """The per-patch running mean and deviation decode() normalizes the prefill with."""
    from timesfm.torch import util
    b = inputs.shape[0]
    x = inputs.reshape(b, -1, p)
    m = masks.reshape(b, -1, p).bool()
    n = torch.zeros(b, dtype=inputs.dtype)
    mu = torch.zeros(b, dtype=inputs.dtype)
    sigma = torch.zeros(b, dtype=inputs.dtype)
    mus, sigmas = [], []
    for i in range(x.shape[1]):
        (n, mu, sigma), _ = util.update_running_stats(n, mu, sigma, x[:, i], m[:, i])
        mus.append(mu)
        sigmas.append(sigma)
    return torch.stack(mus, 1), torch.stack(sigmas, 1)


def relative_max_error(a, e, valid=None):
    diff = (a.double() - e.double()).abs()
    ref = e.double().abs()
    if valid is not None:
        shape = valid.shape + (1,) * (diff.dim() - valid.dim())
        diff = diff * valid.reshape(shape)
        ref = ref * valid.reshape(shape)
    return float(diff.max() / ref.max().clamp(min=1e-6))


PATCH_STAGES = ("normed", "mu", "sigma", "tokens", "layer", "head_point", "pf")


def cpu(t):
    return t.detach().to("cpu", torch.float32).clone().contiguous()


def stack_padded(rows):
    length = max(len(r) for r in rows)
    out = np.full((len(rows), length), np.nan, dtype=np.float32)
    for i, r in enumerate(rows):
        out[i, length - len(r):] = r
    return torch.from_numpy(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="the src/ directory of a google-research/timesfm checkout")
    ap.add_argument("--ckpt", required=True, help="model.safetensors, or a directory holding it")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", default=None, help="comma-separated subset of the cases")
    args = ap.parse_args()
    sys.path.insert(0, str(Path(args.src).resolve()))
    from safetensors.torch import save_file
    from timesfm import configs
    from timesfm.timesfm_2p5 import timesfm_2p5_torch

    torch.set_grad_enabled(False)
    ckpt = Path(args.ckpt)
    if ckpt.is_dir():
        ckpt = ckpt / "model.safetensors"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = timesfm_2p5_torch.TimesFM_2p5_200M_torch(torch_compile=False)
    model.load_checkpoint(str(ckpt), torch_compile=False)
    module = model.model.to("cpu")
    module.device = torch.device("cpu")
    module64 = copy.deepcopy(module).double()
    recorder64 = Recorder(module64)
    recorder = Recorder(module)
    p = module.p
    manifest = {"torch": torch.__version__, "ckpt": str(ckpt), "cases": {}}
    wanted = set(args.cases.split(",")) if args.cases else None

    for case, (h, rows, overrides) in cases().items():
        if wanted and case not in wanted:
            continue
        fc = configs.ForecastConfig(**{**RECOMMENDED, **overrides, "per_core_batch_size": len(rows)})
        model.compile(fc)
        recorder.calls.clear()
        point, quantiles = model.forecast(horizon=h, inputs=[r.copy() for r in rows])
        tensors = {"in.context": cpu(stack_padded(rows)), "point": cpu(torch.from_numpy(point)),
                   "quantiles": cpu(torch.from_numpy(quantiles))}
        horizons = []
        for k, call in enumerate(recorder.calls):
            horizons.append(call.pop("horizon"))
            inputs, masks = call["in.inputs"], call["in.masks"]
            call["mu"], call["sigma"] = running_stats(inputs, masks, p)
            recorder64.calls.clear()
            torch.set_default_dtype(torch.float64)  # decode() allocates its statistics and caches in the default dtype
            try:
                module64.decode(horizons[-1], inputs.double(), masks.bool())
            finally:
                torch.set_default_dtype(torch.float32)
            ref64 = recorder64.calls[0]
            ref64["mu"], ref64["sigma"] = running_stats(inputs.double(), masks, p)
            valid = 1.0 - masks.reshape(masks.shape[0], -1, p)[..., -1]  # a patch counts when its last step is observed
            for name, value in call.items():
                tensors[f"call{k}.{name}"] = cpu(value)
                if not name.startswith("in."):
                    v = valid.double() if name.startswith(PATCH_STAGES) else None
                    tensors[f"call{k}.fp32err.{name}"] = torch.tensor([relative_max_error(value, ref64[name], v)])
        save_file(tensors, out_dir / f"{case}.safetensors")
        cfg = {f.name: getattr(model.forecast_config, f.name) for f in dataclasses.fields(model.forecast_config)}
        manifest["cases"][case] = {"H": h, "config": cfg, "calls": len(recorder.calls), "decode_horizons": horizons,
                                   "ragged": len({len(r) for r in rows}) > 1, "series": len(rows)}
        print(f"{case}: {len(recorder.calls)} decode calls, horizons {horizons}", flush=True)

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))


if __name__ == "__main__":
    main()
