// anf4r: OpenCL C port of Qualcomm's main pass, translated from SPIRV-Cross GLSL by tools/glsl2cl.py.
// SPDX-License-Identifier: GPL-3.0-only
#define LX 8
#define LY 8
#define LZ 1
#include "anf_port.h"

#define _65 ((int)SPECD(2, 1))
#define _174 ((int)SPECD(0, 1))
#define _176 ((int)SPECD(1, 1))
#define _279 ((int)SPECD(4, 0))
#define _280 ((bool)((_279 != 0)))
#define _285 SPECDF(3, 0.999499976634979248046875f)
typedef struct {
    char _m0[];
} _81_83_t;
#define _83 (*(global _81_83_t*)SLOT(2))
typedef struct {
    float2 _m0;
    int _m1;
    int _m2;
    float _m3;
} _25_27_t;
#define _27 (*(constant const _25_27_t*)a->push)
#define _163 TEX(0)
#define _167 SMP(3)
#define _196 TEX(1)
#define _232 TEX(7)
#define _234 SMP(8)
#define _262 TEX(4)
#define _272 TEX(5)
#define _310 TEX(6)

ANF_KERNEL(anf_main)
{
    do
    {
        int2 _17 = ivec2(((uint3)(GID_X, GID_Y, GID_Z)).xy);
        int _22 = _17.x;
        bool _32 = _22 >= _27._m1;
        bool _43;
        if (!_32)
        {
            _43 = _17.y >= _27._m2;
        }
        else
        {
            _43 = _32;
        }
        if (_43)
        {
            break;
        }
        int _54 = _17.y;
        int _75 = ((((_54 / 4) * _65) + (_22 / 4)) * 64) + ((((_54 % 4) * 4) + (_22 % 4)) * 4);
        char _88 = _83._m0[_75];
        char _96 = _83._m0[_75 + 1];
        char _103 = _83._m0[_75 + 2];
        char _111 = _83._m0[_75 + 3];
        float4 _118 = vec4((float)((int)(_88) + 128), (float)((int)(_96) + 128), (float)((int)(_103) + 128), (float)((int)(_111) + 128)) * vec4(0.0039215688593685626983642578125);
        float2 _133 = vec2((float)(_22) - 0.5, (float)(_54) - 0.5) * vec2(0.5);
        float _175 = (float)(_174);
        float _177 = (float)(_176);
        int2 _185 = _17 >> ivec2(1);
        int _194 = (_22 & 1) + (2 * (_54 & 1));
        float4 _199 = texelFetch(_196, _185, 0);
        float _205 = 2.0 * (float)(_185.x);
        float _208 = 2.0 * _199.x;
        float _209 = _205 - _208;
        float _210 = _209 + 0.5;
        float _214 = 2.0 * (float)(_185.y);
        float _217 = 2.0 * _199.y;
        float _218 = _214 - _217;
        float _219 = _218 + 0.5;
        float2 _230 = vec2(_210, _219) + vec2((float)(_194 & 1), (float)(_194 >> 1));
        float _249 = _118.z;
        float3 _259 = (textureLod0(_232, _234, _230 / vec2(2.0 * _175, 2.0 * _177), 0.0).xyz * _249) + (textureLod0(_163, _167, ((g_max(g_floor(vec2(_133.x - _27._m0.x, _133.y + _27._m0.y)), vec2(0.0)) + _118.xy) + vec2(0.5)) / vec2(_175, _177), 0.0).xyz * (1.0 - _249));
        imageStore(_262, _17, vec4(_259, 1.0));
        imageStore(_272, _17, vec4(_118.w, 0.0, 0.0, 0.0));
        if (_280)
        {
            break;
        }
        float3 _288 = g_clamp(_259, vec3(0.0), vec3(_285));
        float3 _306 = (_288 * _27._m3) / vec3(1.0 - g_max(_288.x, g_max(_288.y, _288.z)));
        float3 _309 = _306 * _306;
        imageStore(_310, _17, vec4(_309, 1.0));
        break;
    } while(false);
}
