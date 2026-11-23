#!/usr/bin/env python3
"""Builds the HIP code objects of anf4r's passes (native Windows, runtime/pass.h ANF4R_HIPVK): every kernels/hip/*.cl except
the headers, as <out>/<name>.hsaco for one GPU (the runtime looks a pass up by its SPIR-V file's name). Needs clang and ld.lld
with the AMDGPU target (LLVM 20 or newer: all4r's installer bundles one), found through --llvm, $ANF4R_LLVM or PATH.
The ports of Qualcomm's frame generation passes (AIFF*_cs) are not in the repository: they are derived from the user's own ANF
SDK, so they are made here from its SPIR-V (--assets <dir with spirv/>): spirv-cross --version 450 --vulkan-semantics, then
tools/glsl2cl.py, into <out>/src. Pin spirv-cross to SPIRV-Cross f11ba9f0 (all4r ships a build): other versions may emit
different GLSL.
usage: build_hip.py [--gpu gfx1101] [--out build/hip] [--llvm <dir with clang and ld.lld>] [--assets <ANF assets>]
                    [--spirv-cross <exe>] [names...]
SPDX-License-Identifier: GPL-3.0-only"""
import argparse, glob, os, shutil, subprocess, sys, tempfile

# Qualcomm's frame generation passes anf4r runs (their SPIR-V in the ANF SDK's spirv/ folder)
AIFF = ['AIFFLRUnpackV2CL_cs', 'AIFFLRUpscaleAndQRPackV2_cs', 'AIFFMetaAndLRPackV2_cs', 'AIFFPrepareInputs_cs', 'AIFFPreprocessFusedV2_cs',
        'AIFFPyramidDownscaleFusedV2_cs', 'AIFFQRBlendV2FINT_cs', 'AIFFQRUnpackV2_cs', 'AIFFUnpackMVV2_cs']

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
                        '-Xclang', '-target-feature', '-Xclang', '+extended-image-insts', f'-I{os.path.dirname(src)}', f'-I{os.path.join(ROOT, "kernels", "hip")}', '-c', src, '-o', obj],
                       check=True)
        subprocess.run([lld, '-shared', obj, '-o', out], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default='gfx1101')
    ap.add_argument('--out', default=os.path.join(ROOT, 'build', 'hip'))
    ap.add_argument('--llvm')
    ap.add_argument('--assets', help='ANF SDK assets (spirv/): makes the AIFF*_cs ports')
    ap.add_argument('--spirv-cross', default=os.environ.get('ANF4R_SPIRV_CROSS', 'spirv-cross'))
    ap.add_argument('names', nargs='*')
    a = ap.parse_args()
    clang, lld = tool('clang', a.llvm), tool('ld.lld', a.llvm)
    os.makedirs(a.out, exist_ok=True)
    srcs = sorted(glob.glob(os.path.join(ROOT, 'kernels', 'hip', '*.cl'))) + [os.path.join(ROOT, 'runtime', 'hipvk', 'hipvk_blit.cl')]
    if a.assets:
        if os.path.dirname(a.spirv_cross):   # CreateProcess does not resolve relative paths with forward slashes
            a.spirv_cross = os.path.abspath(a.spirv_cross)
        gen = os.path.join(a.out, 'src')
        os.makedirs(gen, exist_ok=True)
        sys.path.insert(0, os.path.join(ROOT, 'tools'))
        for n in AIFF:
            if a.names and n not in a.names:
                continue
            spv = os.path.join(a.assets, 'spirv', n + '.bin')
            if not os.path.isfile(spv):
                sys.exit(f'build_hip: {spv} missing (the ANF SDK assets)')
            comp, cl = os.path.join(gen, n + '.comp'), os.path.join(gen, n + '.cl')
            subprocess.run([a.spirv_cross, '--version', '450', '--vulkan-semantics', spv, '--output', comp], check=True)
            subprocess.run([sys.executable, '-I', os.path.join(ROOT, 'tools', 'glsl2cl.py'), comp, cl], check=True)
            srcs.append(cl)
    if a.names:
        srcs = [s for s in srcs if os.path.splitext(os.path.basename(s))[0] in a.names]
    for s in srcs:
        name = os.path.splitext(os.path.basename(s))[0]
        if name == 'conv_wmma' and not a.gpu.startswith('gfx11'):
            continue   # RDNA3's WMMA layout only (the runtime uses the scalar convolution elsewhere)
        compile_cl(s, os.path.join(a.out, name + '.hsaco'), a.gpu, clang, lld)
        print(f'{name}.hsaco')


if __name__ == '__main__':
    main()
