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

"""Dump golden inputs, per-stage activations and forecasts of Amazon's Chronos-2 reference.

For each case this writes ``<out>/<case>.safetensors`` holding:
  in.context / in.past_covariates / in.future_covariates   the predict() inputs: (B, V, T) targets
                                                           (NaN-padded on the left for a ragged case),
                                                           (B, Cp, T) past-only covariates and
                                                           (B, Cf, T + H) known covariates
  forecast                                                 the final (B, V, Q, H) quantiles
  call<k>.in.context / .group_ids / .future_covariates     the k-th Chronos2Model.forward call's arguments
  call<k>.attention_mask                                   (R, tokens): 1 where a token is a valid key
  call<k>.loc / .scale                                     the instance norm of each row
  call<k>.embed_context / .embed_future                    the patch embedding of the context and future patches
  call<k>.layer<i>                                         each encoder block's output, (R, tokens, D)
  call<k>.final_norm                                       the encoder's final norm
  call<k>.head                                             the output patch embedding (normalized quantiles)
  call<k>.out                                              forward()'s quantile_preds, (R, Q, patches · P)
  call<k>.fp32err.<stage>                                  how far that stage of the float32 reference is
                                                           from the same call run in float64 (a scalar,
                                                           the relative max error over the valid tokens):
                                                           the noise floor a float32 engine is measured
                                                           against
and ``<out>/manifest.json`` describing every case (H, predict() options, output patches per call).

The Java engine reads these files in its unit tests, so the inputs are identical.

Usage: dump_golden.py --src <chronos-forecasting checkout>/src --ckpt <dir with config.json and model.safetensors> --out golden/
"""

import argparse
import copy
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
    out = torch.full((rows.shape[0], length), float("nan"))
    out[:, length - rows.shape[1]:] = rows
    return out


def cases():
    """name -> (H, [targets (V, T)], [past (Cp, T)] or None, [known (Cf, T + H)] or None, predict kwargs)."""
    out = {}
    out["univariate_short"] = (16, [series(1, 100, 1)], None, None, {})
    out["univariate_long"] = (512, [series(1, 1000, 2, trend=0.01)], None, None, {})
    out["multi_step"] = (1300, [series(1, 900, 3)], None, None, {})
    out["multivariate"] = (64, [series(3, 400, 4)], None, None, {})
    nan_series = series(1, 500, 5)
    nan_series[0, :70] = float("nan")  # a late start
    nan_series[0, 200:260] = float("nan")
    out["with_nans"] = (40, [nan_series], None, None, {})
    out["ragged_batch"] = (48, [series(1, n, 10 + i) for i, n in enumerate((50, 333, 1024))], None, None, {})
    out["covariates"] = (48, [series(1, 600, 20)], [series(2, 600, 21)], [series(2, 648, 22)], {})
    holiday = (torch.arange(800 + 1100) % 168 < 24).float()[None]
    out["covariates_multi_step"] = (1100, [series(1, 800, 23)], [series(1, 800, 24)], [holiday], {})
    out["multitarget_known"] = (40, [series(2, 300, 25)], None, [series(1, 340, 26)], {})
    out["constant"] = (32, [torch.full((1, 200), 5.0)], None, None, {})
    out["long_context"] = (64, [series(1, 9000, 29)], None, None, {})
    out["cross_learning"] = (32, [series(1, 120 + 30 * i, 30 + i) for i in range(4)], None, None, {"cross_learning": True})
    out["many_series"] = (16, [series(1, 64, 100 + i) for i in range(300)], None, None, {})
    return out


def to_inputs(h, targets, past, known):
    """The predict() inputs: (V, T) tensors, or dicts with covariates named so that sorting keeps their order."""
    if past is None and known is None:
        return list(targets)
    out = []
    for i, t in enumerate(targets):
        d = {"target": t, "past_covariates": {}, "future_covariates": {}}
        length = t.shape[-1]
        for j, row in enumerate(past[i] if past is not None else []):
            d["past_covariates"][f"p{j:02d}"] = row
        for j, row in enumerate(known[i] if known is not None else []):
            d["past_covariates"][f"x{j:02d}"] = row[:length]
            d["future_covariates"][f"x{j:02d}"] = row[length:length + h]
        out.append(d)
    return out


class Recorder:
    """Records every Chronos2Model.forward call: its arguments, the stage activations and its output."""

    def __init__(self, model):
        self.model = model
        self.calls = []
        self._current = None
        for i, block in enumerate(model.encoder.block):
            block.register_forward_hook(self._store(f"layer{i}", lambda o: o[0]))
        model.encoder.final_layer_norm.register_forward_hook(self._store("final_norm"))
        model.output_patch_embedding.register_forward_hook(self._store("head"))
        model.input_patch_embedding.register_forward_hook(self._embed)
        model.instance_norm.register_forward_hook(self._norm)
        model.encoder.register_forward_pre_hook(self._mask, with_kwargs=True)
        model.register_forward_pre_hook(self._begin, with_kwargs=True)
        model.register_forward_hook(self._end)

    def _begin(self, module, args, kwargs):
        self._current = {"in.context": kwargs["context"], "in.group_ids": kwargs["group_ids"].float()}
        if kwargs.get("future_covariates") is not None:
            self._current["in.future_covariates"] = kwargs["future_covariates"]
        self._current["num_output_patches"] = kwargs["num_output_patches"]

    def _end(self, module, args, output):
        self._current["out"] = output.quantile_preds
        self.calls.append(self._current)
        self._current = None

    def _mask(self, module, args, kwargs):
        self._current["attention_mask"] = kwargs["attention_mask"].float()

    def _norm(self, module, args, output):
        if "loc" not in self._current:  # the context's; the future covariates reuse it
            loc, scale = output[1]
            self._current["loc"], self._current["scale"] = loc, scale

    def _embed(self, module, args, output):
        self._current["embed_future" if "embed_context" in self._current else "embed_context"] = output

    def _store(self, name, pick=lambda o: o):
        def hook(module, args, output):
            if self._current is not None:
                self._current[name] = pick(output)
        return hook


VALID_TOKENS = ("layer", "final_norm")


def relative_max_error(a, e, mask=None):
    diff = (a.double() - e.double()).abs()
    ref = e.double().abs()
    if mask is not None:
        diff = diff * mask[..., None]
        ref = ref * mask[..., None]
    return float(diff.max() / ref.max().clamp(min=1e-6))


def float64_stages(rec, call):
    """The same forward call on the float64 model (recorded by `rec`): every stage, in float64."""
    rec.calls.clear()
    fc = call.get("in.future_covariates")
    with torch.inference_mode():
        rec.model(context=call["in.context"].double(), group_ids=call["in.group_ids"].long(),
                  future_covariates=fc.double() if fc is not None else None,
                  num_output_patches=call["num_output_patches"])
    return rec.calls[0]


def cpu(t):
    return t.detach().to("cpu", torch.float32).clone().contiguous()


def stack_padded(rows):
    """(B, V, max T) with every series NaN-padded on the left."""
    length = max(r.shape[-1] for r in rows)
    return torch.stack([left_pad(r, length) for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="the src/ directory of a chronos-forecasting checkout")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", default=None, help="comma-separated subset of the cases")
    args = ap.parse_args()
    sys.path.insert(0, str(Path(args.src).resolve()))
    from safetensors.torch import save_file
    from chronos import Chronos2Pipeline

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline = Chronos2Pipeline.from_pretrained(args.ckpt, dtype=torch.float32)
    pipeline.model.eval()
    model64 = copy.deepcopy(pipeline.model).double().eval()
    recorder64 = Recorder(model64)
    recorder = Recorder(pipeline.model)
    manifest = {"torch": torch.__version__, "ckpt": str(args.ckpt), "config": pipeline.model.config.to_dict(), "cases": {}}
    wanted = set(args.cases.split(",")) if args.cases else None

    for case, (h, targets, past, known, kwargs) in cases().items():
        if wanted and case not in wanted:
            continue
        recorder.calls.clear()
        preds = pipeline.predict(to_inputs(h, targets, past, known), prediction_length=h, **kwargs)
        tensors = {"in.context": cpu(stack_padded(targets)), "forecast": cpu(torch.stack(preds))}
        if past is not None:
            tensors["in.past_covariates"] = cpu(torch.stack(past))
        if known is not None:
            tensors["in.future_covariates"] = cpu(torch.stack(known))
        patches = []
        for k, call in enumerate(recorder.calls):
            ref64 = float64_stages(recorder64, call)
            patches.append(call.pop("num_output_patches"))
            mask = call["attention_mask"]
            for name, value in call.items():
                tensors[f"call{k}.{name}"] = cpu(value)
                if not name.startswith("in.") and name != "attention_mask":
                    m = mask.double() if name.startswith(VALID_TOKENS) else None
                    tensors[f"call{k}.fp32err.{name}"] = torch.tensor([relative_max_error(value, ref64[name], m)])
        save_file(tensors, out_dir / f"{case}.safetensors")
        ragged = len({t.shape[-1] for t in targets}) > 1
        manifest["cases"][case] = {"H": h, "kwargs": kwargs, "calls": len(recorder.calls), "output_patches": patches,
                                   "ragged": ragged}
        print(f"{case}: {len(recorder.calls)} forward calls, output patches {patches}")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
