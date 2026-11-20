#!/usr/bin/env python3
"""Derives patches/optiscaler/0001-anf-upscaler.patch and 0002-anf-frame-generation.patch from m4r's MetalFX patches (same OptiScaler
commit fd3a45a): applies them to a scratch clone, renames MetalFX/m4r -> ANF/anf4r (contents and paths), swaps in runtime/anf4r.h, records the diffs.
usage: make_optiscaler_patch.py <OptiScaler git checkout at fd3a45a (read only)> <m4r checkout>"""
import os, shutil, subprocess, sys, tempfile
src, m4r = os.path.realpath(sys.argv[1]), os.path.realpath(sys.argv[2])
here = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
subs = [('MetalFXFlipJitter', 'ANFFlipJitter'), ('MetalFX', 'ANF'), ('metalfx', 'anf'), ('M4RProxy', 'ANFProxy'), ('M4RD3D12', 'ANFD3D12'),
        ('M4R_', 'ANF_'), ('m4r_', 'anf4r_'), ('M4R', 'ANF4R'), ('m4r', 'anf4r')]
def fix(s):
    for a, b in subs: s = s.replace(a, b)
    return (s.replace('\\plans', '\\data').replace('Apple ANF', 'Qualcomm ANF').replace('Apple anf', 'Qualcomm ANF')
             .replace("prepare it with anf4r's install script for", "install the ANF data with scripts/install_anf.py; init failed for"))
def git(d, *a): return subprocess.run(['git', '-C', d, *a], check=True, capture_output=True, text=True).stdout
def build(patches):
    tmp = tempfile.mkdtemp(prefix='anf-opti-'); d = tmp + '/o'
    subprocess.run(['git', 'clone', '-q', src, d], check=True, capture_output=True); git(d, 'checkout', '-q', 'fd3a45a')
    for p in patches: git(d, 'apply', os.path.join(m4r, 'patches/optiscaler', p))
    files = []
    for l in git(d, 'status', '--porcelain', '-uall').splitlines(): files.append(l[3:].strip())
    for f in files:
        p = os.path.join(d, f)
        if os.path.basename(f) == 'm4r.h': continue
        s = open(p, newline='').read(); open(p, 'w', newline='').write(fix(s))
    for f in files:   # rename paths
        nf = fix(f)
        if nf != f:
            os.makedirs(os.path.dirname(os.path.join(d, nf)), exist_ok=True); shutil.move(os.path.join(d, f), os.path.join(d, nf))
    shutil.copy(os.path.join(here, 'runtime/anf4r.h'), os.path.join(d, 'OptiScaler/upscalers/anf/anf4r.h'))
    for dirpath, dirs, fs in os.walk(os.path.join(d, 'OptiScaler')):   # remove emptied dirs
        pass
    return tmp, d
t1, d1 = build(['0001-metalfx-upscaler.patch'])
git(d1, 'add', '-A')
open(os.path.join(here, 'patches/optiscaler/0001-anf-upscaler.patch'), 'w', newline='').write(git(d1, 'diff', '--cached', '--binary'))
git(d1, '-c', 'user.name=x', '-c', 'user.email=x@x', 'commit', '-qm', 'p1')
t2, d2 = build(['0001-metalfx-upscaler.patch', '0002-metalfx-frame-generation.patch'])
for r, ds, fs in os.walk(d2):
    if '.git' in r.split(os.sep): continue
    for f in fs:
        s = os.path.join(r, f); t = os.path.join(d1, os.path.relpath(s, d2)); os.makedirs(os.path.dirname(t), exist_ok=True); shutil.copy2(s, t, follow_symlinks=False)
git(d1, 'add', '-A')
open(os.path.join(here, 'patches/optiscaler/0002-anf-frame-generation.patch'), 'w', newline='').write(git(d1, 'diff', '--cached', '--binary'))
shutil.rmtree(t1); shutil.rmtree(t2)
print('wrote patches/optiscaler/0001-anf-upscaler.patch, 0002-anf-frame-generation.patch')
