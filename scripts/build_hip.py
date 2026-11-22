#!/usr/bin/env python3
"""Builds the HIP code objects of anf4r's passes (native Windows, runtime/pass.h ANF4R_HIPVK): every kernels/hip/*.cl except
the headers, as <out>/<name>.hsaco for one GPU (the runtime looks a pass up by its SPIR-V file's name). The ports of Qualcomm's
frame generation passes are kernels/hip/AIFF*_cs.cl. Needs clang and ld.lld with the AMDGPU target (LLVM 20 or newer: m4r's
installer bundles one), found through --llvm, $ANF4R_LLVM or PATH.
usage: build_hip.py [--gpu gfx1101] [--out build/hip] [--llvm <dir with clang and ld.lld>] [names...]
SPDX-License-Identifier: GPL-3.0-only"""
import argparse, glob, os, shutil, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def tool(name, llvm):
    for d in ([llvm] if llvm else []) + ([os.environ['ANF4R_LLVM']] if os.environ.get('ANF4R_LLVM') else []):
        for n in (name + '.exe', name):
            if os.path.isfile(os.path.join(d, n)):
                return os.path.join(d, n)
    p = shutil.which(name)
    if not p:
        sys.exit(f'build_hip: {name} not found (--llvm or ANF4R_LLVM)')
    return p


def compile_cl(src, out, gpu, clang, lld):
    with tempfile.TemporaryDirectory() as d:
        obj = os.path.join(d, 'k.o')
        subprocess.run([clang, '-x', 'cl', '-cl-std=CL2.0', '--target=amdgcn-amd-amdhsa', f'-mcpu={gpu}', '-nogpulib', '-O3', '-cl-single-precision-constant',
                        '-Xclang', '-target-feature', '-Xclang', '+extended-image-insts', f'-I{os.path.dirname(src)}', '-c', src, '-o', obj],
                       check=True)
        subprocess.run([lld, '-shared', obj, '-o', out], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default='gfx1101')
    ap.add_argument('--out', default=os.path.join(ROOT, 'build', 'hip'))
    ap.add_argument('--llvm')
    ap.add_argument('names', nargs='*')
    a = ap.parse_args()
    clang, lld = tool('clang', a.llvm), tool('ld.lld', a.llvm)
    os.makedirs(a.out, exist_ok=True)
    srcs = sorted(glob.glob(os.path.join(ROOT, 'kernels', 'hip', '*.cl'))) + [os.path.join(ROOT, 'runtime', 'hipvk', 'hipvk_blit.cl')]
    if a.names:
        srcs = [s for s in srcs if os.path.splitext(os.path.basename(s))[0] in a.names]
    for s in srcs:
        name = os.path.splitext(os.path.basename(s))[0]
        compile_cl(s, os.path.join(a.out, name + '.hsaco'), a.gpu, clang, lld)
        print(f'{name}.hsaco')


if __name__ == '__main__':
    main()
