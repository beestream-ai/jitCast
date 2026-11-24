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

"""Dump golden inputs, per-stage activations and forecasts of The Forecasting Company's t0 reference.

For each case this writes ``<out>/<case>.safetensors`` holding:
  s<i>.target / .past / .future     series i as jitcast's TimeSeries holds it: (Vt, T), (Vp, T) and
                                    (Vf, T + H), NaN where missing (.past / .future absent without them)
  quantiles                         predict()'s output, (target rows, H, Q), the series' targets in order
  call<k>.in.variates / .mask       the k-th forward window (T0Forecaster._predict_step's argument), (V, W):
  call<k>.in.group / .type          values, MaskType, group ids and VariateType, as float32
  call<k>.scaled / .loc / .scale    the causal scaler's output and statistics, (V, W)
  call<k>.embed                     the patch encoder's output, (V, P, D)
  call<k>.layer<i>                  each transformer layer's output, (V, P, D)
  call<k>.hidden                    the transformer's output after its final RMSNorm, (V, P, D)
  call<k>.decoded                   the decoder's raw output, (V, P, patch · Q_trained)
  call<k>.head                      the quantile head's output (scaled), (V, P, patch, Q_trained)
  call<k>.out                       _predict_step's output in data space, (V, horizon, Q_trained)
  call<k>.fp32err.<stage>           how far that stage of the float32 reference is from the same window
                                    run in float64 (a scalar, the relative max error over the cells and
                                    patches that are not padding): the noise floor a float32 engine is
                                    measured against
and ``<out>/manifest.json`` describing every case (H, predict options, each call's horizon).

Usage: dump_golden.py --src <tfc-t0 checkout> --ckpt <dir with model.safetensors and config.json> --out golden/
"""

import argparse
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


def counts(length, seed):
    g = np.random.default_rng(seed)
    t = np.arange(length)
    return g.poisson(3.0 + 2.0 * np.sin(2 * math.pi * t / 7)).astype(np.float32)


def calendar(length, period, phase=0.0):
    t = np.arange(length, dtype=np.float64)
    return np.sin(2 * math.pi * (t + phase) / period).astype(np.float32)


def uni(values):
    return {"target": np.asarray(values, dtype=np.float32)[None, :]}


# predict()'s keyword defaults, except the levels: jitcast returns the checkpoint's levels by default.
DEFAULTS = dict(trim_leading_missing=False, symmetric_averaging=False, clip_nonnegative=False,
                snap_integers=False, drop_constant_covariates=True)


def cases():
    """name -> (H, [series dicts], predict overrides). A series dict holds target (Vt, T), past (Vp, T), future (Vf, T + H)."""
    out = {}
    out["univariate_short"] = (16, [uni(series(100, 1))], {})
    out["univariate_long"] = (512, [uni(series(2000, 2, trend=0.01))], {})
    out["rollout"] = (1300, [uni(series(700, 3))], {"quantile_levels": [0.1, 0.5, 0.9]})
    out["rollout_tails"] = (1090, [uni(series(400, 4)), uni(series(260, 13))],
                            {"quantile_levels": [0.005, 0.25, 0.5, 0.75, 0.995]})
    out["tails"] = (64, [uni(series(300, 14))], {"quantile_levels": [0.001, 0.02, 0.33, 0.5, 0.98, 0.999]})
    nans = series(500, 5)
    nans[:70] = np.nan  # a late start
    nans[200:260] = np.nan  # a gap
    out["with_nans"] = (40, [uni(nans)], {})
    out["trim_leading"] = (40, [uni(nans)], {"trim_leading_missing": True})
    out["ragged_batch"] = (48, [uni(series(n, 10 + i)) for i, n in enumerate((50, 333, 1024))], {})
    out["multivariate"] = (64, [{"target": np.stack([series(600, 20), series(600, 21, level=3.0), series(600, 22) * 2])}], {})
    t, h = 480, 96
    out["covariates"] = (h, [{
        "target": series(t, 30)[None, :] + 3 * calendar(t, 24)[None, :],
        "past": np.stack([series(t, 31), series(t, 32, level=-4.0)]),
        "future": np.stack([calendar(t + h, 24), calendar(t + h, 168, 5.0), np.full(t + h, 2.0, dtype=np.float32)]),
    }], {})
    out["covariates_batch"] = (32, [
        {"target": series(300, 40)[None, :], "future": np.stack([calendar(332, 12)])},
        {"target": np.stack([series(200, 41), series(200, 42)]), "past": np.stack([series(200, 43)])},
        uni(series(150, 44)),
    ], {})
    out["covariates_rollout"] = (1060, [{
        "target": series(300, 45)[None, :] + 2 * calendar(300, 24)[None, :],
        "future": np.stack([calendar(1360, 24)]),
    }], {"quantile_levels": [0.1, 0.5, 0.9]})
    mixed = series(400, 50)
    mixed[100:130] = np.nan
    out["adjustments"] = (48, [uni(counts(300, 51)), uni(mixed), uni(np.abs(series(250, 52)))],
                          {"trim_leading_missing": True, "symmetric_averaging": True, "clip_nonnegative": True,
                           "snap_integers": True})
    out["constant"] = (32, [uni(np.full(200, 5.0, dtype=np.float32))], {})
    out["negative"] = (64, [uni(series(400, 6, level=-20.0))], {})
    out["near_zero"] = (64, [uni(np.abs(series(300, 7, level=0.0)) * 0.01)], {})
    out["large_scale"] = (64, [uni(series(600, 12, level=1e6) * 1e3)], {})
    out["long_context"] = (128, [uni(series(4096, 8))], {})
    out["many_series"] = (16, [uni(series(64 + (i % 7) * 13, 100 + i)) for i in range(40)], {})
    return out


def native(entry, h, T0):
    """One jitcast series as t0's native TimeSeries, width T + H: targets and past covariates (HISTORICAL)
    withheld over the horizon, known-future covariates valid wherever they are not NaN."""
    from t0 import MaskType, VariateType

    target = entry["target"]
    t = target.shape[1]
    past = entry.get("past", np.zeros((0, t), np.float32))
    future = entry.get("future", np.zeros((0, t + h), np.float32))[:, : t + h]
    rows, masks, types = [], [], []
    for values, kind in ((target, VariateType.TARGET), (past, VariateType.HISTORICAL)):
        for row in values:
            rows.append(np.concatenate([np.nan_to_num(row, nan=0.0), np.zeros(h, np.float32)]))
            masks.append(np.concatenate([np.where(np.isnan(row), MaskType.MISSING, MaskType.VALID),
                                         np.full(h, MaskType.WITHHELD)]))
            types.append(kind)
    for row in future:
        rows.append(np.nan_to_num(row, nan=0.0))
        masks.append(np.where(np.isnan(row), MaskType.MISSING, MaskType.VALID))
        types.append(VariateType.FUTURE)
    n, width = len(rows), t + h
    return T0(
        variates=torch.from_numpy(np.stack(rows).astype(np.float32)),
        mask=torch.from_numpy(np.stack(masks).astype(np.int8)),
        group_ids=torch.zeros((n, width), dtype=torch.int64),
        variate_type=torch.tensor(types, dtype=torch.int64)[:, None].expand(n, width).contiguous(),
    )


class Recorder:
    """Records each forward window of a model and the stages of its pass."""

    def __init__(self, model):
        self.model = model
        self.calls = []
        self.current = None
        step = model._predict_step
        scale_input = model.scaler.scale_input

        def wrapped_step(window, horizon):
            self.current = {"horizon": horizon, "window": window}
            out = step(window, horizon)
            self.current["out"] = out
            self.calls.append(self.current)
            self.current = None
            return out

        def wrapped_scale(series):
            scaled, ls = scale_input(series)
            if self.current is not None:
                self.current.update({"scaled": scaled.variates, "loc": ls.loc, "scale": ls.scale})
            return scaled, ls

        model._predict_step = wrapped_step
        model.scaler.scale_input = wrapped_scale
        model.patch_encoder.register_forward_hook(self._store("embed"))
        for i, layer in enumerate(model.transformer.layers):
            layer.register_forward_hook(self._store(f"layer{i}"))
        model.transformer.out_norm.register_forward_hook(self._store("hidden"))
        model.decoder.register_forward_hook(self._store("decoded"))
        model.head.register_forward_hook(self._store("head"))

    def _store(self, name):
        def hook(module, args, output):
            if self.current is not None:
                self.current[name] = output
        return hook


def relative_max_error(a, e, valid):
    diff = (a.double() - e.double()).abs()
    ref = e.double().abs()
    shape = valid.shape + (1,) * (diff.dim() - valid.dim())
    v = valid.reshape(shape).expand_as(diff)
    diff = torch.where(v, diff, torch.zeros_like(diff))
    ref = torch.where(v, ref, torch.zeros_like(ref))
    if torch.isnan(diff).any():
        return float("inf")
    return float(diff.max() / ref.max().clamp(min=1e-6))


PATCH_STAGES = ("embed", "layer", "hidden", "decoded", "head")
CELL_STAGES = ("scaled", "loc", "scale")


def cpu(t):
    return t.detach().to("cpu", torch.float32).clone().contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="a theforecastingcompany/tfc-t0 checkout")
    ap.add_argument("--ckpt", required=True, help="the directory holding model.safetensors and config.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", default=None, help="comma-separated subset of the cases")
    args = ap.parse_args()
    sys.path.insert(0, str(Path(args.src).resolve()))
    from safetensors.torch import save_file
    from t0 import MaskType, T0Forecaster, TimeSeries

    torch.set_grad_enabled(False)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = str(Path(args.ckpt).resolve())
    model = T0Forecaster.from_pretrained(ckpt).eval().to("cpu")
    model64 = T0Forecaster.from_pretrained(ckpt).eval().to("cpu").double()
    recorder = Recorder(model)
    recorder64 = Recorder(model64)
    trained = [float(q) for q in model.config.quantile_levels]
    p = model.patch_size
    manifest = {"torch": torch.__version__, "ckpt": str(args.ckpt), "patch": p, "trained_levels": trained, "cases": {}}
    wanted = set(args.cases.split(",")) if args.cases else None

    for case, (h, entries, overrides) in cases().items():
        if wanted and case not in wanted:
            continue
        options = {**DEFAULTS, "quantile_levels": trained, **overrides}
        batch = TimeSeries.batch([native(e, h, TimeSeries) for e in entries])
        recorder.calls.clear()
        forecast = model.predict(batch, h, options["quantile_levels"],
                                 **{k: v for k, v in options.items() if k != "quantile_levels"})
        tensors = {"quantiles": cpu(forecast.quantiles)}
        for i, e in enumerate(entries):
            for key in ("target", "past", "future"):
                if key in e:
                    tensors[f"s{i}.{key}"] = torch.from_numpy(np.ascontiguousarray(e[key], dtype=np.float32))
        horizons = []
        for k, call in enumerate(recorder.calls):
            horizons.append(call.pop("horizon"))
            window = call.pop("window")
            tensors[f"call{k}.in.variates"] = cpu(window.variates)
            tensors[f"call{k}.in.mask"] = cpu(window.mask)
            tensors[f"call{k}.in.group"] = cpu(window.group_ids)
            tensors[f"call{k}.in.type"] = cpu(window.variate_type)
            window64 = dataclasses.replace(window, variates=window.variates.double())
            recorder64.calls.clear()
            recorder64.model._predict_step(window64, horizons[-1])
            ref64 = recorder64.calls[0]
            mask = window.mask
            rows, width = mask.shape
            cell_valid = window.group_ids >= 0
            patch_valid = (mask.reshape(rows, width // p, p) != MaskType.PAD).any(-1)
            for name, value in call.items():
                tensors[f"call{k}.{name}"] = cpu(value)
                if name.startswith(PATCH_STAGES):
                    valid = patch_valid
                elif name in CELL_STAGES:
                    valid = cell_valid
                else:
                    valid = torch.ones(value.shape[:1], dtype=torch.bool)
                tensors[f"call{k}.fp32err.{name}"] = torch.tensor([relative_max_error(value, ref64[name], valid)])
        save_file(tensors, out_dir / f"{case}.safetensors")
        manifest["cases"][case] = {"H": h, "options": options, "series": len(entries), "calls": len(recorder.calls),
                                   "horizons": horizons}
        floor = max(float(v) for key, v in tensors.items() if ".fp32err." in key) if recorder.calls else 0.0
        print(f"{case}: {len(recorder.calls)} forward calls, horizons {horizons}, worst fp32 floor {floor:.2e}", flush=True)

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))


if __name__ == "__main__":
    main()
