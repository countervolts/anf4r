// anf4r: OpenCL C port of Qualcomm's main pass, translated from SPIRV-Cross GLSL by tools/glsl2cl.py.
// SPDX-License-Identifier: GPL-3.0-only
#define LX 8
#define LY 8
#define LZ 1
#include "anf_port.h"

#define _269 SPECDF(0, 0.999499976634979248046875f)
typedef struct {
    int _m0;
    int _m1;
    float _m2;
    float _m3;
} _58_60_t;
#define _60 (*(constant const _58_60_t*)a->push)
#define _84 TEX(0)
#define _188 TEX(1)
#define _225 TEX(2)
#define _229 SMP(3)
#define _258 TEX(4)
#define _295 TEX(5)

ANF_KERNEL(anf_main)
{
    local float3 _75[10][10];
    do
    {
        int2 _17 = ivec2(((uint3)(GID_X, GID_Y, GID_Z)).xy);
        int2 _28 = (ivec2(((uint3)(WG_X, WG_Y, WG_Z)).xy) * ivec2(8)) - ivec2(1);
        int _34 = (int)(LIDX);
        for (int _328 = _34; _328 < 100; )
        {
            int _47 = _328 % 10;
            int _49 = _328 / 10;
            _75[_49][_47] = texelFetch(_84, g_clamp(_28 + ivec2(_47, _49), ivec2(0), ivec2(_60._m0 - 1, _60._m1 - 1)), 0).xyz;
            _328 += 64;
            continue;
        }
        LDS_BARRIER();
        int _98 = _17.x;
        bool _101 = _98 >= _60._m0;
        bool _110;
        if (!_101)
        {
            _110 = _17.y >= _60._m1;
        }
        else
        {
            _110 = _101;
        }
        if (_110)
        {
            break;
        }
        int2 _120 = ivec2(((uint3)(LID_X, LID_Y, LID_Z)).xy) + ivec2(1);
        int _124 = _120.y;
        int _126 = _120.x;
        int _329;
        float3 _330;
        float3 _331;
        _331 = _75[_124][_126];
        _330 = _75[_124][_126];
        _329 = -1;
        float3 _334;
        float3 _335;
        for (; _329 <= 1; _331 = _335, _330 = _334, _329++)
        {
            _335 = _331;
            _334 = _330;
            for (int _332 = -1; _332 <= 1; )
            {
                int _154 = _124 + _329;
                int _158 = _126 + _332;
                _335 = g_max(_335, _75[_154][_158]);
                _334 = g_min(_334, _75[_154][_158]);
                _332++;
                continue;
            }
        }
        int2 _174 = _17 >> ivec2(1);
        int _184 = (_98 & 1) + (2 * (_17.y & 1));
        float4 _191 = texelFetch(_188, _174, 0);
        float3 _255 = g_mix(_75[_124][_126], g_clamp(textureLod0(_225, _229, vec2(((2.0 * ((float)(_174.x) - _191.x)) + 0.5) + (float)(_184 & 1), ((2.0 * ((float)(_174.y) - _191.y)) + 0.5) + (float)(_184 >> 1)) / vec2((float)(_60._m0), (float)(_60._m1)), 0.0).xyz, _330, _331), vec3(_60._m2));
        imageStore(_258, _17, vec4(_255, 1.0));
        float3 _272 = g_clamp(_255, vec3(0.0), vec3(_269));
        float3 _291 = (_272 * _60._m3) / vec3(1.0 - g_max(_272.x, g_max(_272.y, _272.z)));
        imageStore(_295, _17, vec4(_291 * _291, 1.0));
        break;
    } while(false);
}
