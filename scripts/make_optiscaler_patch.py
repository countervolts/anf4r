#!/usr/bin/env python3
"""Derives patches/optiscaler/0001-anf-upscaler.patch from m4r's MetalFX upscaler patch (same OptiScaler commit fd3a45a):
applies m4r's patch to a clean checkout, renames MetalFX/m4r -> ANF/anf4r, swaps in anf4r.h, and records the diff.
usage: make_optiscaler_patch.py <clean OptiScaler checkout at fd3a45a> <m4r checkout> """
import os, re, subprocess, sys, shutil
import tempfile
src, m4r = os.path.realpath(sys.argv[1]), os.path.realpath(sys.argv[2])
_tmp = tempfile.mkdtemp(prefix='anf-opti-')
subprocess.run(['git', 'clone', '-q', src, _tmp + '/o'], check=True, capture_output=True)   # work on a scratch clone; the build tree is untouched
opti = _tmp + '/o'
here = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
def git(*a, check=True): return subprocess.run(['git', '-C', opti, *a], check=check, capture_output=True, text=True)
git('checkout', '-q', 'fd3a45a')
assert git('status', '--porcelain').stdout.strip() == ''
git('apply', os.path.join(m4r, 'patches/optiscaler/0001-metalfx-upscaler.patch'))
subs = [('MetalFXFeatureDx12', 'ANFFeatureDx12'), ('MetalFXFlipJitter', 'ANFFlipJitter'), ('MetalFX', 'ANF'), ('metalfx', 'anf'),
        ('M4RProxy', 'ANFProxy'), ('M4RD3D12', 'ANFD3D12'), ('M4R_', 'ANF_'), ('m4r_', 'anf4r_'), ('M4R', 'ANF4R'), ('m4r', 'anf4r')]
changed = git('status', '--porcelain').stdout.splitlines()
files = []
for l in changed:
    p = l[3:].strip()
    if p.endswith('/'):
        for r, _, fs in os.walk(os.path.join(opti, p)): files += [os.path.relpath(os.path.join(r, f), opti) for f in fs]
    else: files.append(p)
for f in files:
    path = os.path.join(opti, f)
    if os.path.basename(f) == 'm4r.h': continue                      # replaced below
    s = open(path, newline='').read()
    for a, b in subs: s = s.replace(a, b)
    s = s.replace('\\plans', '\\data').replace('Apple ANF', 'Qualcomm ANF').replace('Apple anf', 'Qualcomm ANF')
    s = s.replace("prepare it with anf4r's install script for", "install the ANF data with scripts/install_anf.py; init failed for")
    s = s.replace('"{} (install the ANF data with scripts/install_anf.py; init failed for {}x{} -> {}x{})"', '"{} (ANF SR is fixed 2.0x; init failed for {}x{} -> {}x{})"')
    open(path, 'w', newline='').write(s)
# renames
d = os.path.join(opti, 'OptiScaler/upscalers')
os.rename(os.path.join(d, 'metalfx'), os.path.join(d, 'anf'))
for old, new in [('M4R_D3D12.h', 'ANF_D3D12.h'), ('M4R_Proxy.h', 'ANF_Proxy.h'), ('MetalFXFeature_Dx12.cpp', 'ANFFeature_Dx12.cpp'), ('MetalFXFeature_Dx12.h', 'ANFFeature_Dx12.h')]:
    os.rename(os.path.join(d, 'anf', old), os.path.join(d, 'anf', new))
os.remove(os.path.join(d, 'anf', 'm4r.h'))
shutil.copy(os.path.join(here, 'runtime/anf4r.h'), os.path.join(d, 'anf', 'anf4r.h'))
# paths inside the vcxproj/includes refer to the old names
for r, _, fs in os.walk(opti):
    if '.git' in r.split(os.sep): continue
    for f in fs:
        if f.endswith(('.h', '.cpp', '.vcxproj', '.ini', '.txt', '.md')):
            p = os.path.join(r, f)
            try: s = open(p, newline='').read()
            except Exception: continue
            t = s.replace('upscalers\\metalfx\\', 'upscalers\\anf\\').replace('upscalers/metalfx/', 'upscalers/anf/').replace('MetalFXFeature_Dx12', 'ANFFeature_Dx12').replace('M4R_D3D12', 'ANF_D3D12').replace('M4R_Proxy', 'ANF_Proxy').replace('"m4r.h"', '"anf4r.h"').replace('m4r.h', 'anf4r.h')
            if t != s: open(p, 'w', newline='').write(t)
git('add', '-A')
out = git('diff', '--cached', '--binary').stdout
open(os.path.join(here, 'patches/optiscaler/0001-anf-upscaler.patch'), 'w', newline='').write(out)
shutil.rmtree(_tmp)
print('wrote patches/optiscaler/0001-anf-upscaler.patch', len(out.splitlines()), 'lines')
