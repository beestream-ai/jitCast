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

"""Dump golden inputs, per-stage activations and forecasts of Mistral's Timestral reference.

For each case this writes ``<out>/<case>.safetensors`` holding:
  in.context / in.past_covariates / in.known_covariates     the forecast() inputs, (B, V, T) etc.
  forecast                                                  the final (B, V, H, Q) quantiles
  call<k>.in.inputs / .observed / .group_ids / .role_ids    the k-th predict_block call's arguments
  call<k>.in.known_future / .known_future_mask
  call<k>.scaled / .loc / .scale                            the scaler (after the known-covariate scaling)
  call<k>.embed                                             the patch embedding of the encoder tokens
  call<k>.layer<i>                                          each transformer layer's output, [Enc ; Dec] tokens
  call<k>.out_norm                                          the final norm
  call<k>.head                                              the normalized quantiles of every decoder token
  call<k>.out                                               predict_block's raw-space quantiles
  call<k>.fp32err.<stage>                                   how far that stage of the float32 reference is
                                                            from the same call run in float64 (a scalar,
                                                            the relative max error): the noise floor a
                                                            float32 engine is measured against
and ``<out>/manifest.json`` describing every case (H, options, horizon patches per call).

The Java engine reads these files in its unit tests, so the inputs are identical.

Usage: dump_golden.py --src <timestral checkout>/src --ckpt <dir with config.json and model.safetensors> --out golden/
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch


def series(n, length, seed, trend=0.0, season=24, level=10.0):
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(length, dtype=torch.float32)
    rows = []
    for i in range(n):
        walk = torch.randn(length, generator=g).cumsum(0) * 0.3
        s = 2.0 * torch.sin(2 * math.pi * t / season + i)
        rows.append(level + walk + s + trend * t)
    return torch.stack(rows)


def left_pad(rows, length):
    """Left-pads (V, T) rows with NaN to `length`, the way a ragged batch is passed to forecast()."""
    out = torch.full((rows.shape[0], length), float("nan"))
    out[:, length - rows.shape[1]:] = rows
    return out


def cases():
    """name -> (H, context (B, V, T), past (B, C, T) or None, known (B, C, T + H) or None, forecast kwargs)."""
    out = {}
    out["univariate_short"] = (16, series(1, 100, 1)[None], None, None, {})
    out["univariate_refine"] = (300, series(1, 700, 2, trend=0.01)[None], None, None, {})
    out["multi_block"] = (1200, series(1, 1000, 3)[None], None, None, {})
    out["multivariate"] = (64, series(3, 400, 4)[None], None, None, {})
    nan_series = series(1, 500, 5)
    nan_series[0, :70] = float("nan")  # a late start: the first observed patch is not patch 0
    nan_series[0, 200:260] = float("nan")
    out["with_nans"] = (40, nan_series[None], None, None, {})
    lengths = (50, 333, 1024)
    out["ragged_batch"] = (48, torch.stack([left_pad(series(1, n, 10 + i), max(lengths)) for i, n in enumerate(lengths)]),
                           None, None, {})
    out["covariates"] = (48, series(1, 600, 20)[None], series(2, 600, 21)[None], series(2, 648, 22)[None], {})
    holiday = (torch.arange(800 + 1100) % 168 < 24).float()[None, None]
    out["covariates_multi_block"] = (1100, series(1, 800, 23)[None], series(1, 800, 24)[None], holiday, {})
    out["multitarget_known"] = (40, series(2, 300, 25)[None], None, series(1, 340, 26)[None], {})
    counts = torch.poisson(torch.full((1, 1, 400), 2.0), generator=torch.Generator().manual_seed(27))
    out["nonnegative"] = (64, counts, None, None, {})
    out["constant"] = (32, torch.full((1, 1, 200), 5.0), None, None, {})
    out["no_symmetric"] = (40, series(1, 300, 28)[None], None, None, {"symmetric_averaging": False})
    out["overlong_context"] = (64, series(1, 33000, 29)[None], None, None, {})
    return out


class Recorder:
    """Records every predict_block call: its arguments, the stage activations and its output."""

    def __init__(self, model):
        self.model = model
        self.calls = []
        self._current = None
        self._layers = []
        for i, layer in enumerate(model.transformer.layers):
            layer.register_forward_hook(self._store(f"layer{i}"))
        model.transformer.out_norm.register_forward_hook(self._store("out_norm"))
        model.representation.patch_proj.register_forward_hook(self._store("embed"))
        model.head.register_forward_hook(self._store("head"))
        scale = model.scale
        predict = model.predict_block
        known = model._known_cov_scaling

        def scale_hook(inputs, observed):
            out = scale(inputs, observed)
            if self._current is not None and "scaled" not in self._current:
                self._put(out)
            return out

        def known_hook(*args):
            out = known(*args)
            if self._current is not None:
                self._put(out)  # replaces the causal statistics on the known-covariate rows
            return out

        def predict_hook(inputs, observed, group_ids, horizon_patches, role_ids=None, known_future=None,
                         known_future_mask=None):
            self._current = {"in.inputs": inputs, "in.observed": observed.float(), "in.group_ids": group_ids.float()}
            if role_ids is not None:
                self._current["in.role_ids"] = role_ids.float()
            if known_future is not None:
                self._current["in.known_future"] = known_future
                self._current["in.known_future_mask"] = known_future_mask.float()
            out = predict(inputs, observed, group_ids, horizon_patches, role_ids, known_future, known_future_mask)
            self._current["out"] = out
            self._current["horizon_patches"] = horizon_patches
            self.calls.append(self._current)
            self._current = None
            return out

        model.scale = scale_hook
        model._known_cov_scaling = known_hook
        model.predict_block = predict_hook

    def _put(self, scaled_loc_scale):
        scaled, loc, scale = scaled_loc_scale
        self._current["scaled"], self._current["loc"], self._current["scale"] = scaled, loc, scale

    def _store(self, name):
        def hook(module, args, output):
            if self._current is not None:
                self._current[name] = output
        return hook


def relative_max_error(a, e):
    return float((a.double() - e.double()).abs().max() / e.double().abs().max().clamp(min=1e-6))


def float64_stages(rec, call):
    """The same predict_block call on the float64 model (recorded by `rec`): every stage, in float64."""
    rec.calls.clear()
    model64 = rec.model
    args = [call["in.inputs"].double(), call["in.observed"].bool(), call["in.group_ids"].long(), call["horizon_patches"]]
    role = call.get("in.role_ids")
    kf = call.get("in.known_future")
    kfm = call.get("in.known_future_mask")
    with torch.inference_mode():
        model64.predict_block(*args, role.long() if role is not None else None, kf.double() if kf is not None else None,
                              kfm.bool() if kfm is not None else None)
    return rec.calls[0]


def cpu(t):
    return t.detach().to("cpu", torch.float32).clone().contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="the src/ directory of a timestral checkout")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", default=None, help="comma-separated subset of the cases")
    args = ap.parse_args()
    sys.path.insert(0, str(Path(args.src).resolve()))
    from safetensors.torch import save_file
    from timestral import TimestralForecaster

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    from timestral.loading import load_model
    forecaster = TimestralForecaster.from_pretrained(args.ckpt, device="cpu")
    recorder64 = Recorder(load_model(args.ckpt, dtype=torch.float64))
    recorder = Recorder(forecaster.model)
    manifest = {"torch": torch.__version__, "ckpt": str(args.ckpt), "config": forecaster.config.to_dict(), "cases": {}}
    wanted = set(args.cases.split(",")) if args.cases else None

    for case, (h, context, past, known, kwargs) in cases().items():
        if wanted and case not in wanted:
            continue
        recorder.calls.clear()
        fc = forecaster.forecast(context, h, past_covariates=past, known_covariates=known, **kwargs)
        tensors = {"in.context": cpu(context), "forecast": cpu(fc.quantiles)}
        if past is not None:
            tensors["in.past_covariates"] = cpu(past)
        if known is not None:
            tensors["in.known_covariates"] = cpu(known)
        horizons = []
        for k, call in enumerate(recorder.calls):
            ref64 = float64_stages(recorder64, call)
            horizons.append(call.pop("horizon_patches"))
            for name, value in call.items():
                tensors[f"call{k}.{name}"] = cpu(value)
                if not name.startswith("in."):
                    tensors[f"call{k}.fp32err.{name}"] = torch.tensor([relative_max_error(value, ref64[name])])
        save_file(tensors, out_dir / f"{case}.safetensors")
        manifest["cases"][case] = {"H": h, "kwargs": kwargs, "calls": len(recorder.calls), "horizon_patches": horizons}
        print(f"{case}: {len(recorder.calls)} predict_block calls, horizon patches {horizons}")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
