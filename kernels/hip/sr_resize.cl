// anf4r: OpenCL C port of Qualcomm's main pass, translated from SPIRV-Cross GLSL by tools/glsl2cl.py.
// SPDX-License-Identifier: GPL-3.0-only
#define LX 8
#define LY 8
#define LZ 1
#include "anf_port.h"

typedef struct {
    int2 _m0;
    int2 _m1;
} _74_76_t;
#define _76 (*(constant const _74_76_t*)a->push)
#define _172 TEX(0)
#define _193 TEX(1)

ANF_KERNEL(anf_main)
{
    do
    {
        int2 _68 = ivec2(((uint3)(GID_X, GID_Y, GID_Z)).xy);
        bool _81 = _68.x >= _76._m1.x;
        bool _91;
        if (!_81)
        {
            _91 = _68.y >= _76._m1.y;
        }
        else
        {
            _91 = _81;
        }
        if (_91)
        {
            break;
        }
        float2 _113 = (((vec2(_68) + vec2(0.5)) * vec2(_76._m0)) / vec2(_76._m1)) - vec2(0.5);
        int2 _117 = ivec2(g_floor(_113));
        float2 _122 = _113 - vec2(_117);
        float _127 = _122.x;
        float _217 = _127 * _127;
        float _220 = _217 * _127;
        float _226 = 0.5 * _127;
        float4 _124 = vec4((((-0.5) * _220) + _217) - _226, ((1.5 * _220) - (2.5 * _217)) + 1.0, (((-1.5) * _220) + (2.0 * _217)) + _226, 0.5 * (_220 - _217));
        float _132 = _122.y;
        float _254 = _132 * _132;
        float _257 = _254 * _132;
        float _263 = 0.5 * _132;
        float4 _129 = vec4((((-0.5) * _257) + _254) - _263, ((1.5 * _257) - (2.5 * _254)) + 1.0, (((-1.5) * _257) + (2.0 * _254)) + _263, 0.5 * (_257 - _254));
        int _289;
        float3 _290;
        _290 = vec3(0.0);
        _289 = 0;
        float3 _293;
        for (; _289 < 4; _290 = _293, _289++)
        {
            _293 = _290;
            for (int _291 = 0; _291 < 4; )
            {
                _293 += (texelFetch(_172, g_clamp(_117 + ivec2(_291 - 1, _289 - 1), ivec2(0), _76._m0 - ivec2(1)), 0).xyz * (_124[_291] * _129[_289]));
                _291++;
                continue;
            }
        }
        imageStore(_193, _68, vec4(g_max(_290, vec3(0.0)), 1.0));
        break;
    } while(false);
}
