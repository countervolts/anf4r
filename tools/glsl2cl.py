#!/usr/bin/env python3
"""Translates SPIRV-Cross GLSL (spirv-cross --version 450 --vulkan-semantics <pass>_cs.bin) of a compute pass into an OpenCL C
port for anf4r's HIP backend (kernels/hip/anf_port.h): resources become argument-block slots, specialisation constants and push
constants read from the block, shared variables become kernel locals, GLSL types / constructors / built-ins map onto anf_port.h's
overloads. The result is meant to be checked (and, where a construct is not handled, finished) by hand.
usage: glsl2cl.py <in.comp> <out.cl> [entry]   (entry: the runtime's entry point name, default main_cs -> kernel anf_main_cs)
SPDX-License-Identifier: GPL-3.0-only"""
import re, sys

VEC = {f'{p}vec{n}': f'{t}{n}' for p, t in (('', 'float'), ('i', 'int'), ('u', 'uint'), ('b', 'int')) for n in (2, 3, 4)}
VEC.update({'int8_t': 'char', 'uint8_t': 'uchar', 'int16_t': 'short', 'uint16_t': 'ushort', 'float16_t': 'half'})
BUILTINS = ['clamp', 'min', 'max', 'mix', 'floor', 'ceil', 'fract', 'abs', 'dot', 'roundEven', 'trunc', 'fma', 'sqrt', 'exp2', 'log2',
            'pow', 'exp', 'sign', 'step', 'smoothstep']


def ctype(t):
    return VEC.get(t, t)


def atomics_by_address(body):
    """atomicMin(lvalue, v) -> atomicMin(&(lvalue), v): GLSL passes the memory, C needs its address"""
    out, i = [], 0
    for m in re.finditer(r'\batomic(Min|Max|Add|And|Or|Exchange)\(', body):
        if m.start() < i:
            continue
        j, depth = m.end(), 0
        while True:
            c = body[j]
            if c in '([': depth += 1
            elif c in ')]': depth -= 1
            elif c == ',' and depth == 0: break
            j += 1
        out.append(body[i:m.end()] + '&(' + body[m.end():j] + ')')
        i = j
    return ''.join(out) + body[i:]


def main():
    src, out = sys.argv[1], sys.argv[2]
    entry = sys.argv[3] if len(sys.argv) > 3 else 'main_cs'
    text = open(src).read()
    head, shared, body_lines = [], [], []
    tex_kind = {}   # texture / image name -> '', 'I', 'U'
    lx = ly = lz = 1
    lines = text.split('\n')
    i = 0
    in_main = False
    depth = 0
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if in_main:
            body_lines.append(ln)
            depth += ln.count('{') - ln.count('}')
            if depth == 0 and '}' in ln:
                in_main = False
            i += 1
            continue
        if not s or s.startswith('#version') or s.startswith('#extension') or s.startswith('spirv_instruction'):
            i += 1; continue
        m = re.match(r'layout\(local_size_x = (\d+), local_size_y = (\d+), local_size_z = (\d+)\) in;', s)
        if m:
            lx, ly, lz = m.groups(); i += 1; continue
        m = re.match(r'layout\(constant_id = (\d+)\) const (\w+) (\w+) = (.*);', s)
        if m:   # a constant the host leaves unset keeps its GLSL default, as in Vulkan
            cid, t, name, dv = m.groups()
            dv = dv.strip()
            if t == 'float':
                head.append(f'#define {name} SPECDF({cid}, {dv}f)' if re.match(r'^-?[\d.]+(e-?\d+)?$', dv) else f'#define {name} SPECDF({cid}, {dv})')
            elif t == 'bool':
                head.append(f'#define {name} (SPECD({cid}, {1 if dv == "true" else 0}) != 0)')
            else:
                head.append(f'#define {name} (({t})SPECD({cid}, {dv.rstrip("u")}))')
            i += 1; continue
        m = re.match(r'const (\w+) (\w+) = (.*);$', s)
        if m:
            t, name, expr = m.groups()
            head.append(f'#define {name} (({ctype(t)})({expr}))' if t in ('int', 'uint', 'float', 'bool') else f'#define {name} ({expr})')
            i += 1; continue
        m = re.match(r'layout\((.*)\) (?:(readonly|writeonly|restrict|coherent) )*(buffer|uniform) (\w+)$', s)
        if m:   # a block: buffer (binding) or push constant
            lay, _, kind, tname = m.groups()
            members = []
            i += 1
            while not lines[i].strip().startswith('}'):
                if lines[i].strip() not in ('{', ''):
                    members.append(lines[i].strip())
                i += 1
            inst = re.match(r'\}\s*(\w+);', lines[i].strip()).group(1)
            fields = '\n'.join('    ' + re.sub(r'^(\w+)', lambda q: ctype(q.group(1)), mm) for mm in members)
            head.append(f'typedef struct {{\n{fields}\n}} {tname}_t;')
            if 'push_constant' in lay:
                head.append(f'#define {inst} (*(constant const {tname}_t*)a->push)')
            else:
                b = re.search(r'binding = (\d+)', lay).group(1)
                head.append(f'#define {inst} (*(global {tname}_t*)SLOT({b}))')
            i += 1; continue
        m = re.match(r'layout\(set = 0, binding = (\d+)(?:, \w+)?\) uniform (?:(?:readonly|writeonly) )*(\w*)(texture2D|image2D|sampler) (\w+);', s)
        if m:
            b, pre, kind, name = m.groups()
            head.append(f'#define {name} ' + ('SMP' if kind == 'sampler' else 'TEX') + f'({b})')
            if kind != 'sampler':
                tex_kind[name] = {'i': 'I', 'u': 'U'}.get(pre, '')
            i += 1; continue
        m = re.match(r'shared (\w+) (\w+)(\[.*\])?;', s)
        if m:
            t, name, arr = m.groups()
            shared.append(f'    local {ctype(t)} {name}{arr or ""};')
            i += 1; continue
        if s == 'void main()':
            in_main = True
            depth = 0
            i += 1
            body_lines.append('{')
            depth = 1
            i += 1   # skip the opening brace line
            continue
        head.append('// UNHANDLED: ' + ln)
        i += 1

    body = '\n'.join(body_lines)
    # built-in variables
    body = body.replace('gl_GlobalInvocationID', '((uint3)(GID_X, GID_Y, GID_Z))')
    body = body.replace('gl_LocalInvocationID', '((uint3)(LID_X, LID_Y, LID_Z))')
    body = body.replace('gl_WorkGroupID', '((uint3)(WG_X, WG_Y, WG_Z))')
    body = body.replace('gl_LocalInvocationIndex', 'LIDX')
    # resources
    body = re.sub(r'textureLod\(sampler2D\((\w+), (\w+)\), ', r'textureLod0(\1, \2, ', body)
    body = re.sub(r'texture\(sampler2D\((\w+), (\w+)\), ', r'texture0(\1, \2, ', body)
    for name, k in tex_kind.items():
        body = re.sub(rf'\btexelFetch\({name}\b', f'texelFetch{k}({name}', body)
        body = re.sub(rf'\bimageStore\({name}\b', f'imageStore{k}({name}', body)
        body = re.sub(rf'\bimageLoad\({name}\b', f'imageLoad{k}({name}', body)
    body = body.replace('barrier();', 'LDS_BARRIER();')
    body = atomics_by_address(body)
    # types in declarations, casts, built-ins, swizzles
    body = re.sub(r'\b(b?[iu]?vec[234])\b(?=\s+[A-Za-z_]\w*\s*(=|;|\[))', lambda q: ctype(q.group(1)), body)
    body = re.sub(r'\bprecise ', '', body)   # separate multiplies and adds are never contracted here
    body = re.sub(r'\b(int8_t|uint8_t|int16_t|uint16_t|float16_t)\b', lambda q: VEC[q.group(1)], body)
    body = re.sub(r'\b(float|int|uint|bool|char|uchar|short|ushort|half)\(', r'(\1)(', body)
    body = re.sub(r'\binversesqrt\(', 'g_rsqrt(', body)
    body = re.sub(r'\b(' + '|'.join(BUILTINS) + r')\(', r'g_\1(', body)
    body = re.sub(r'\.([rgba]{1,4})\b', lambda q: '.' + q.group(1).translate(str.maketrans('rgba', 'xyzw')), body)
    out_text = [f'// anf4r: OpenCL C port of Qualcomm\'s {entry} pass, translated from SPIRV-Cross GLSL by tools/glsl2cl.py.',
                '// SPDX-License-Identifier: GPL-3.0-only', f'#define LX {lx}', f'#define LY {ly}', f'#define LZ {lz}', '#include "anf_port.h"', '']
    out_text += head + ['', f'ANF_KERNEL(anf_{entry})'] + [body_lines and '{'] + shared + body.split('\n')[1:]
    open(out, 'w').write('\n'.join(out_text) + '\n')


if __name__ == '__main__':
    main()
