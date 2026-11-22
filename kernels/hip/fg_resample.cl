// anf4r: OpenCL C port of Qualcomm's main pass, translated from SPIRV-Cross GLSL by tools/glsl2cl.py.
// SPDX-License-Identifier: GPL-3.0-only
#define LX 8
#define LY 8
#define LZ 1
#include "anf_port.h"

typedef struct {
    float2 _m0;
    float2 _m1;
    int2 _m2;
} _25_27_t;
#define _27 (*(constant const _25_27_t*)a->push)
#define _62 TEX(0)
#define _77 TEX(1)
#define _90 TEX(2)
#define _114 TEX(3)

ANF_KERNEL(anf_main)
{
    do
    {
        int2 _17 = ivec2(((uint3)(GID_X, GID_Y, GID_Z)).xy);
        bool _32 = _17.x >= _27._m2.x;
        bool _42;
        if (!_32)
        {
            _42 = _17.y >= _27._m2.y;
        }
        else
        {
            _42 = _32;
        }
        if (_42)
        {
            break;
        }
        float2 _57 = (vec2(_17) + vec2(0.5)) / vec2(_27._m2);
        imageStore(_90, _17, vec4(texelFetch(_62, g_min(ivec2(_57 * vec2(textureSize(_62, 0))), (textureSize(_62, 0) - ivec2(1))), 0).x, 0.0, 0.0, 0.0));
        imageStore(_114, _17, vec4((texelFetch(_77, g_min(ivec2(_57 * vec2(textureSize(_77, 0))), (textureSize(_77, 0) - ivec2(1))), 0).xy * _27._m0) / _27._m1, 0.0, 0.0));
        break;
    } while(false);
}
