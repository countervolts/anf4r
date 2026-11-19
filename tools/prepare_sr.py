#!/usr/bin/env python3
"""anf4r: build an SR plan from extracted ANF assets, like m4r's prepare_*.py.

A plan is a directory the runtime loads: the chosen QMFNet model's per-jitter weights (int8), biases (int32)
and per-output-channel scales (float), plus a manifest describing the layer graph and the input/output resolution.
Qualcomm's weights are copied unchanged; nothing here is Qualcomm code.

usage: prepare_sr.py --assets vendor/assets --out plans/sr-<outWxH>-<inWxH> --in WxH [--model invpow2|prod7p1]
"""
import argparse, json, os, shutil, glob

LAYERS = [  # QMFNet V2 (prod-7p1): name, in_ch, out_ch, stride
    ("cnn1", 20, 32, 2), ("cnn2-0", 32, 32, 1), ("cnn2-1", 32, 32, 1), ("cnn3", 32, 64, 1),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default="vendor/assets")
    ap.add_argument("--out", required=True)
    ap.add_argument("--in", dest="insize", required=True, help="WxH render size")
    ap.add_argument("--model", default="invpow2", choices=["invpow2", "prod7p1"])
    a = ap.parse_args()
    iw, ih = (int(x) for x in a.insize.lower().split("x"))
    md = os.path.join(a.assets, "modelzoo",
                      "aiff-sr-v2s-prod-7p1-invpow2" if a.model == "invpow2" else "aiff-sr-v2s-prod-7p1")
    if not os.path.isdir(md): raise SystemExit("model dir not found: " + md)
    phases = sorted(int(p) for p in os.listdir(os.path.join(md, "cnn1")) if p.isdigit())
    os.makedirs(a.out, exist_ok=True)
    manifest = {"technique": "sr", "model": os.path.basename(md), "input": [iw, ih], "output": [iw*2, ih*2],
                "jitter_phases": len(phases), "layers": []}
    def pick(d, sub, ext):
        for f in os.listdir(d):
            if sub in f and f.endswith(ext): return os.path.join(d, f)
        return None
    for (name, ic, oc, stride) in LAYERS:
        manifest["layers"].append({"name": name, "in_ch": ic, "out_ch": oc, "stride": stride, "pad": 1, "relu": True})
        avail=sorted(int(p) for p in os.listdir(os.path.join(md,name)) if p.isdigit())
        for ph in avail:
            src = os.path.join(md, name, str(ph))
            dst = os.path.join(a.out, "wts", name, str(ph)); os.makedirs(dst, exist_ok=True)
            for sub, ext in (("weight", ".raw"), ("bias", ".raw"), ("scales", ".json")):
                s = pick(src, sub, ext)
                if s: shutil.copy(s, os.path.join(dst, {"weight": "weights.raw", "bias": "bias.raw", "scales": "scales.json"}[sub]))
    json.dump(manifest, open(os.path.join(a.out, "plan.json"), "w"), indent=1)
    nbytes = sum(os.path.getsize(f) for f in glob.glob(os.path.join(a.out, "**"), recursive=True) if os.path.isfile(f))
    print(f"wrote plan {a.out}: {len(LAYERS)} layers x {len(phases)} jitter phases, {nbytes//1024} KiB")

if __name__ == "__main__":
    main()
