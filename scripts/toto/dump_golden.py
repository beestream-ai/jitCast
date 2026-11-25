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

"""Dump golden inputs, per-stage activations and forecasts of Datadog's Toto 2.0 reference.

The reference is run the way its GluonTS predictor runs it (``Toto2GluonTSModel`` with the default
``Toto2GluonTSModelConfig``: forward-filled targets, missing-value masks, the short-patch scaler fallback,
the real-space quantile cap and a single forward pass), with the instance splitter's context trimming and
left padding reproduced here.

For each case this writes ``<out>/<case>.safetensors`` holding:
  series<i>.target / .past          the raw inputs of series i, (V, T) and (V_p, T), NaN where missing
  quantiles                         the forecasts, (B, V, H, 9), levels 0.1..0.9
  in.target / in.observed           Toto2GluonTSModel.forward's past_target and past_observed_target,
                                    (B, V, C), after imputation, trimming and padding; observed as 0/1
  in.is_pad                         its past_is_pad, (B, C)
  in.past / in.past_observed        its past_feat_dynamic_real and mask, (B, V_p, C) (absent without)
  loc / scale                       the causal scaler's statistics over the forward's whole input,
                                    (B, V + V_p, C + horizon patches * 32), before the short-patch fallback
  tokens                            the patch embedding, (B, V + V_p, S, D)
  layer<i>                          each transformer layer's output, (B, V + V_p, S, D)
  out_norm                          the final norm's output, (B, V + V_p, S, D)
  head                              the output head on the forecast patches, normalized and asinh-space,
                                    (9, B, V + V_p, horizon patches, 32)
  fp32err.<stage>                   how far that stage of the float32 reference is from the same case run
                                    in float64 (a scalar, the relative max error): the noise floor a float32
                                    engine is measured against
and ``<out>/manifest.json`` describing every case (H, context length, shapes).

Usage: dump_golden.py --ckpt <dir with config.json and model.safetensors> --out golden/
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

P = 32


def walk(length, seed, level=10.0, season=24, trend=0.0, noise=0.3):
    g = np.random.default_rng(seed)
    t = np.arange(length, dtype=np.float64)
    return (level + np.cumsum(g.standard_normal(length)) * noise + 2.0 * np.sin(2 * math.pi * t / season)
            + trend * t).astype(np.float32)


def holes(x, seed, fraction, lead=0):
    x = x.copy()
    g = np.random.default_rng(seed)
    x[..., g.random(x.shape[-1]) < fraction] = np.nan
    x[..., :lead] = np.nan
    return x


def cases():
    """name -> (H, context_length, [(target (V, T), past (V_p, T) or None)])."""
    c = {}
    c["uni"] = (96, 4096, [(walk(512, 1)[None], None)])
    c["ragged"] = (64, 4096, [(walk(n, 10 + n, level=5 + n / 100)[None], None) for n in (100, 300, 512, 777)])
    c["missing"] = (48, 4096, [(holes(walk(600, 2)[None], 3, 0.15, lead=70), None),
                               (holes(walk(600, 4, season=7)[None], 5, 0.4), None)])
    c["long"] = (1000, 4096, [(walk(1024, 6, season=168, trend=0.01)[None], None),
                              (walk(800, 7, season=12)[None], None)])
    c["trim"] = (200, 1024, [(walk(5000, 8, season=96)[None], None), (walk(1500, 9)[None], None)])
    c["short"] = (24, 4096, [(walk(20, 11)[None], None), (np.full((1, 45), 3.0, np.float32), None),
                             (walk(9, 12)[None], None)])
    c["scale"] = (60, 4096, [(walk(400, 13, level=1e6, noise=3e3)[None], None),
                             (walk(400, 14, level=1e-3, noise=1e-5)[None], None),
                             (np.abs(walk(400, 15, level=0.0, noise=1.0))[None], None)])
    base = walk(700, 16, season=24)
    mv = np.stack([base, 0.5 * base + walk(700, 17, level=3), holes(walk(700, 18, season=12)[None], 19, 0.1)[0]])
    c["multi"] = (96, 4096, [(mv, None), (mv[:, 100:] * 1.3, None)])
    c["past"] = (96, 4096, [(walk(640, 20)[None], np.stack([walk(640, 21), holes(walk(640, 22)[None], 23, 0.2)[0]])),
                            (walk(400, 24)[None], np.stack([walk(400, 25), walk(400, 26, season=7)]))])
    return c


def ffill(x):
    """Forward-fill NaN along the last axis; leading NaN become 0 (GluonTS with ffill imputation)."""
    out = x.copy()
    for r in range(out.shape[0]):
        last = 0.0
        for t in range(out.shape[1]):
            if np.isnan(out[r, t]):
                out[r, t] = last
            else:
                last = out[r, t]
    return out


def assemble(items, h, context_length):
    """The GluonTS input transform plus TFTInstanceSplitter, for one batch."""
    ctx = context_length - (math.ceil(h / P) - 1) * P
    past_len = max(P, (ctx // P) * P)
    b, v = len(items), items[0][0].shape[0]
    vp = 0 if items[0][1] is None else items[0][1].shape[0]
    target = np.zeros((b, v, past_len), np.float32)
    observed = np.zeros((b, v, past_len), np.float32)
    is_pad = np.ones((b, past_len), np.float32)
    past = np.zeros((b, vp, past_len), np.float32)
    past_obs = np.zeros((b, vp, past_len), np.float32)
    for i, (tg, pc) in enumerate(items):
        n = min(tg.shape[1], past_len)
        target[i, :, past_len - n:] = ffill(tg)[:, -n:]
        observed[i, :, past_len - n:] = ~np.isnan(tg[:, -n:])
        is_pad[i, past_len - n:] = 0
        if vp:
            past[i, :, past_len - n:] = np.nan_to_num(pc[:, -n:], nan=0.0)
            past_obs[i, :, past_len - n:] = ~np.isnan(pc[:, -n:])
    return target, observed, is_pad, (past, past_obs) if vp else None


class Recorder:
    def __init__(self, model):
        self.model, self.out, self.handles = model, {}, []
        m = model
        self.handles.append(m.scaler.register_forward_hook(self._scaler))
        self.handles.append(m.patch_proj.register_forward_hook(self._tokens))
        for i, layer in enumerate(m.transformer.layers):
            self.handles.append(layer.register_forward_hook(self._layer(i)))
        self.handles.append(m.transformer.out_norm.register_forward_hook(self._store("out_norm")))
        self.handles.append(m.output_head.register_forward_hook(self._store("head")))

    def _scaler(self, module, args, output):
        self.out["loc"], self.out["scale"] = output[1].detach().clone(), output[2].detach().clone()

    def _tokens(self, module, args, output):
        # forecast() embeds the context and the forecast patches in two calls
        prev = self.out.get("tokens")
        self.out["tokens"] = output.detach().clone() if prev is None else torch.cat([prev, output.detach()], -2)

    def _store(self, name):
        def hook(module, args, output):
            self.out[name] = output.detach().clone()
        return hook

    def _layer(self, i):
        def hook(module, args, output):
            self.out[f"layer{i}"] = output.detach().clone()
        return hook

    def shaped(self, b, v):
        """Layer outputs back in (B, V, S, D): time layers run on (B*V, S, D), variate layers on (B*S, V, D)."""
        out = dict(self.out)
        dec = self.model.transformer
        for i in range(len(dec.layers)):
            x = out[f"layer{i}"]
            if dec._if_variate_layer(i):
                out[f"layer{i}"] = x.reshape(b, x.shape[0] // b, v, -1).permute(0, 2, 1, 3).contiguous()
            else:
                out[f"layer{i}"] = x.reshape(b, v, x.shape[-2], -1)
        return out


def relative_max_error(a, e):
    diff = (a.double() - e.double()).abs()
    return float(diff.max() / e.double().abs().max().clamp(min=1e-6))


def cpu(t):
    return t.detach().to("cpu", torch.float32).clone().contiguous()


def run(gts, inputs, dtype):
    target, observed, is_pad, past = inputs
    t = lambda a: torch.from_numpy(a).to(dtype)
    kw = {}
    if past is not None:
        kw = {"past_feat_dynamic_real": t(past[0]), "past_observed_feat_dynamic_real": t(past[1]).bool()}
    with torch.no_grad():
        (q,), _, _ = gts(t(target), t(observed).bool(), t(is_pad).bool(), **kw)
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="a directory with config.json and model.safetensors")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", default=None, help="comma-separated subset of the cases")
    args = ap.parse_args()

    from safetensors.torch import save_file
    from toto2 import Toto2GluonTSModel, Toto2GluonTSModelConfig, Toto2Model

    torch.manual_seed(0)
    model32 = Toto2Model.from_pretrained(args.ckpt).eval()
    model64 = Toto2Model.from_pretrained(args.ckpt).eval().double()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"torch": torch.__version__, "ckpt": str(args.ckpt), "cases": {}}
    selected = None if args.cases is None else set(args.cases.split(","))
    for case, (h, context_length, items) in cases().items():
        if selected is not None and case not in selected:
            continue
        v = items[0][0].shape[0]
        vp = 0 if items[0][1] is None else items[0][1].shape[0]
        inputs = assemble(items, h, context_length)
        stages = {}
        for name, model, dtype in (("32", model32, torch.float32), ("64", model64, torch.float64)):
            cfg = Toto2GluonTSModelConfig(prediction_length=h, context_length=context_length, target_dim=v,
                                          past_feat_dynamic_real_dim=vp)
            gts = Toto2GluonTSModel(model, cfg).eval()
            rec = Recorder(model)
            q = run(gts, inputs, dtype)
            for hd in rec.handles:
                hd.remove()
            st = rec.shaped(len(items), v + vp)
            st["quantiles"] = q.reshape(len(items), v, h, 9)
            stages[name] = st
        tensors = {}
        for i, (tg, pc) in enumerate(items):
            tensors[f"series{i}.target"] = torch.from_numpy(np.ascontiguousarray(tg))
            if pc is not None:
                tensors[f"series{i}.past"] = torch.from_numpy(np.ascontiguousarray(pc))
        target, observed, is_pad, past = inputs
        tensors["in.target"], tensors["in.observed"] = torch.from_numpy(target), torch.from_numpy(observed)
        tensors["in.is_pad"] = torch.from_numpy(is_pad)
        if past is not None:
            tensors["in.past"], tensors["in.past_observed"] = torch.from_numpy(past[0]), torch.from_numpy(past[1])
        errs = {}
        for name, value in stages["32"].items():
            tensors[name] = cpu(value)
            if name != "quantiles":
                errs[name] = relative_max_error(value, stages["64"][name])
                tensors[f"fp32err.{name}"] = torch.tensor([errs[name]])
        save_file(tensors, out_dir / f"{case}.safetensors")
        manifest["cases"][case] = {"H": h, "context_length": context_length, "series": len(items), "targets": v,
                                   "past": vp, "C": int(target.shape[-1]),
                                   "quantiles_fp32err": relative_max_error(stages["32"]["quantiles"],
                                                                           stages["64"]["quantiles"])}
        print(f"{case}: H={h} B={len(items)} V={v}+{vp} C={target.shape[-1]} worst fp32err "
              f"{max(errs.values()):.1e} quantiles {manifest['cases'][case]['quantiles_fp32err']:.1e}")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
