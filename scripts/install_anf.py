#!/usr/bin/env python3
"""Assembles the anf4r runtime folder that OptiScaler loads (<game>/anf4r/): anf4r.dll + data/{shaders,spirv,modelzoo}.
Assembles from your own ANF model data in data/ (see NOTICE.md; it must not be redistributed).
usage: install_anf.py --out <game dir or any dir> [--dll build/win/anf4r.dll] [--sr-model aiff-sr-v2s-prod-7p1-invpow2]"""
import argparse, os, shutil, glob
ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--dll', default='build/win/anf4r.dll')
ap.add_argument('--sr-model', default='aiff-sr-v2s-prod-7p1-invpow2')
a = ap.parse_args()
root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
dst = os.path.join(a.out, 'anf4r'); data = os.path.join(dst, 'data')
if os.path.isdir(dst): shutil.rmtree(dst)
os.makedirs(os.path.join(data, 'shaders')); os.makedirs(os.path.join(data, 'spirv')); os.makedirs(os.path.join(data, 'modelzoo'))
shutil.copy(os.path.join(root, a.dll), os.path.join(dst, 'anf4r.dll'))
for f in glob.glob(os.path.join(root, 'build', '*.spv')): shutil.copy(f, os.path.join(data, 'shaders'))
for f in glob.glob(os.path.join(root, 'data/spirv/*.bin')): shutil.copy(f, os.path.join(data, 'spirv'))
for m in (a.sr_model, 'aiff-frc-lr-v2-prod-8', 'aiff-frc-qr-v2-prod-8'):
    shutil.copytree(os.path.join(root, 'data/modelzoo', m), os.path.join(data, 'modelzoo', m))
print('installed to', dst, '(%d shaders)' % len(os.listdir(os.path.join(data, 'shaders'))))
