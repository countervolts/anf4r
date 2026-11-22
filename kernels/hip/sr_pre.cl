// anf4r: OpenCL C port of Qualcomm's main pass, translated from SPIRV-Cross GLSL by tools/glsl2cl.py.
// SPDX-License-Identifier: GPL-3.0-only
#define LX 8
#define LY 8
#define LZ 1
#include "anf_port.h"

#define _150 ((int)SPECD(0, 1))
#define _157 ((int)SPECD(1, 1))
typedef struct {
    uint _m0[];
} _430_432_t;
#define _432 (*(global _430_432_t*)SLOT(6))
typedef struct {
    float _m0;
    float2 _m1;
    int _m2;
} _74_76_t;
#define _76 (*(constant const _74_76_t*)a->push)
#define _164 TEX(1)
#define _179 TEX(2)
#define _225 TEX(0)
#define _369 TEX(3)
#define _373 SMP(5)
#define _387 TEX(4)
#define _401 TEX(7)
#define _411 TEX(8)

ANF_KERNEL(anf_main)
{
    local float _188[10][10];
    local float2 _199[10][10];
    do
    {
        int2 _95 = ivec2(((uint3)(GID_X, GID_Y, GID_Z)).xy);
        int2 _106 = (ivec2(((uint3)(WG_X, WG_Y, WG_Z)).xy) * ivec2(8)) - ivec2(1);
        int _112 = (int)(LIDX);
        for (int _1116 = _112; _1116 < 100; _1116 += 64)
        {
            int _124 = _1116 % 10;
            int _126 = _1116 / 10;
            int2 _131 = _106 + ivec2(_124, _126);
            int _138 = _131.x;
            bool _139 = _138 >= 0;
            bool _145;
            if (_139)
            {
                _145 = _131.y >= 0;
            }
            else
            {
                _145 = _139;
            }
            bool _152;
            if (_145)
            {
                _152 = _138 < _150;
            }
            else
            {
                _152 = _145;
            }
            bool _159;
            if (_152)
            {
                _159 = _131.y < _157;
            }
            else
            {
                _159 = _152;
            }
            float _1145;
            float2 _1150;
            if (_159)
            {
                float4 _168 = texelFetch(_164, _131, 0);
                float _169 = _168.x;
                float _1149;
                if (_76._m2 != 0)
                {
                    _1149 = 1.0 - _169;
                }
                else
                {
                    _1149 = _169;
                }
                _1150 = texelFetch(_179, _131, 0).xy;
                _1145 = _1149;
            }
            else
            {
                _1150 = vec2(0.0);
                _1145 = -1.0;
            }
            _188[_126][_124] = _1145;
            _199[_126][_124] = _1150;
        }
        LDS_BARRIER();
        int _212 = _95.x;
        bool _213 = _212 >= _150;
        bool _220;
        if (!_213)
        {
            _220 = _95.y >= _157;
        }
        else
        {
            _220 = _213;
        }
        if (_220)
        {
            break;
        }
        float3 _588 = g_sqrt(g_max(texelFetch(_225, _95, 0).xyz, vec3(0.0)));
        float3 _603 = _588 * (1.0 / (g_max(_588.x, g_max(_588.y, _588.z)) + _76._m0));
        int2 _238 = ivec2(((uint3)(LID_X, LID_Y, LID_Z)).xy) + ivec2(1);
        int _241 = _238.y;
        int _243 = _238.x;
        int _249 = _241 - 1;
        int _252 = _243 - 1;
        float2 _1118;
        _1118 = _199[_249][_252];
        float _1142;
        float2 _1159;
        int _1117 = -1;
        float _1143 = _188[_249][_252];
        for (; _1117 <= 1; _1118 = _1159, _1117++, _1143 = _1142)
        {
            _1142 = _1143;
            _1159 = _1118;
            float _1167;
            float2 _1160;
            for (int _1140 = -1; _1140 <= 1; _1142 = _1167, _1140++, _1159 = _1160)
            {
                int _285 = _241 + _1117;
                int _289 = _243 + _1140;
                bool _294 = _188[_285][_289] > _1142;
                if (_294)
                {
                    _1160 = _199[_285][_289];
                }
                else
                {
                    _1160 = _1159;
                }
                _1167 = _294 ? _188[_285][_289] : _1142;
            }
        }
        float2 _317 = _1118 * _76._m1;
        float _323 = 2.0 * (float)(_212);
        float _326 = 2.0 * _317.x;
        float _327 = _323 - _326;
        float _328 = _327 + 0.5;
        int _330 = _95.y;
        float _332 = 2.0 * (float)(_330);
        float _335 = 2.0 * _317.y;
        float _336 = _332 - _335;
        float _337 = _336 + 0.5;
        float2 _338 = vec2(_328, _337);
        float _340 = (float)(_150);
        float _342 = (float)(_157);
        float2 _344 = vec2(2.0 * _340, 2.0 * _342);
        float4 _367[4];
        float _385[4];
        for (int _1119 = 0; _1119 < 4; )
        {
            float2 _363 = _338 + vec2((float)(_1119 & 1), (float)(_1119 >> 1));
            float2 _379 = _363 / _344;
            _367[_1119] = textureLod0(_369, _373, _379, 0.0);
            _385[_1119] = textureLod0(_387, _373, _379, 0.0).x;
            _1119++;
            continue;
        }
        float _405 = _603.x;
        imageStore(_401, _95, vec4(_405, _603.yz, 0.0));
        imageStore(_411, _95, vec4(_317, 0.0, 0.0));
        uint _428 = (uint)((_330 * _150) + _212) * 8u;
        float _608 = _405 * 255.0;
        float _1120;
        if (_608 >= 0.0)
        {
            _1120 = g_floor(_608 + 0.5);
        }
        else
        {
            _1120 = g_ceil(_608 - 0.5);
        }
        float _632 = _603.y * 255.0;
        float _1121;
        if (_632 >= 0.0)
        {
            _1121 = g_floor(_632 + 0.5);
        }
        else
        {
            _1121 = g_ceil(_632 - 0.5);
        }
        float _656 = _603.z * 255.0;
        float _1122;
        if (_656 >= 0.0)
        {
            _1122 = g_floor(_656 + 0.5);
        }
        else
        {
            _1122 = g_ceil(_656 - 0.5);
        }
        float _680 = _188[_241][_243] * 255.0;
        float _1123;
        if (_680 >= 0.0)
        {
            _1123 = g_floor(_680 + 0.5);
        }
        else
        {
            _1123 = g_ceil(_680 - 0.5);
        }
        _432._m0[_428] = ((((uint)(g_clamp(_1120, 0.0, 255.0)) ^ 128u) | (((uint)(g_clamp(_1121, 0.0, 255.0)) ^ 128u) << (uint)(8))) | (((uint)(g_clamp(_1122, 0.0, 255.0)) ^ 128u) << (uint)(16))) | (((uint)(g_clamp(_1123, 0.0, 255.0)) ^ 128u) << (uint)(24));
        float _704 = _385[0] * 255.0;
        float _1124;
        if (_704 >= 0.0)
        {
            _1124 = g_floor(_704 + 0.5);
        }
        else
        {
            _1124 = g_ceil(_704 - 0.5);
        }
        float _728 = _385[1] * 255.0;
        float _1125;
        if (_728 >= 0.0)
        {
            _1125 = g_floor(_728 + 0.5);
        }
        else
        {
            _1125 = g_ceil(_728 - 0.5);
        }
        float _752 = _385[2] * 255.0;
        float _1126;
        if (_752 >= 0.0)
        {
            _1126 = g_floor(_752 + 0.5);
        }
        else
        {
            _1126 = g_ceil(_752 - 0.5);
        }
        float _776 = _385[3] * 255.0;
        float _1127;
        if (_776 >= 0.0)
        {
            _1127 = g_floor(_776 + 0.5);
        }
        else
        {
            _1127 = g_ceil(_776 - 0.5);
        }
        _432._m0[_428 + 1u] = ((((uint)(g_clamp(_1124, 0.0, 255.0)) ^ 128u) | (((uint)(g_clamp(_1125, 0.0, 255.0)) ^ 128u) << (uint)(8))) | (((uint)(g_clamp(_1126, 0.0, 255.0)) ^ 128u) << (uint)(16))) | (((uint)(g_clamp(_1127, 0.0, 255.0)) ^ 128u) << (uint)(24));
        float _800 = _367[0].x * 255.0;
        float _1128;
        if (_800 >= 0.0)
        {
            _1128 = g_floor(_800 + 0.5);
        }
        else
        {
            _1128 = g_ceil(_800 - 0.5);
        }
        float _824 = _367[1].x * 255.0;
        float _1129;
        if (_824 >= 0.0)
        {
            _1129 = g_floor(_824 + 0.5);
        }
        else
        {
            _1129 = g_ceil(_824 - 0.5);
        }
        float _848 = _367[2].x * 255.0;
        float _1130;
        if (_848 >= 0.0)
        {
            _1130 = g_floor(_848 + 0.5);
        }
        else
        {
            _1130 = g_ceil(_848 - 0.5);
        }
        float _872 = _367[3].x * 255.0;
        float _1131;
        if (_872 >= 0.0)
        {
            _1131 = g_floor(_872 + 0.5);
        }
        else
        {
            _1131 = g_ceil(_872 - 0.5);
        }
        _432._m0[_428 + 2u] = ((((uint)(g_clamp(_1128, 0.0, 255.0)) ^ 128u) | (((uint)(g_clamp(_1129, 0.0, 255.0)) ^ 128u) << (uint)(8))) | (((uint)(g_clamp(_1130, 0.0, 255.0)) ^ 128u) << (uint)(16))) | (((uint)(g_clamp(_1131, 0.0, 255.0)) ^ 128u) << (uint)(24));
        float _896 = _367[0].y * 255.0;
        float _1132;
        if (_896 >= 0.0)
        {
            _1132 = g_floor(_896 + 0.5);
        }
        else
        {
            _1132 = g_ceil(_896 - 0.5);
        }
        float _920 = _367[1].y * 255.0;
        float _1133;
        if (_920 >= 0.0)
        {
            _1133 = g_floor(_920 + 0.5);
        }
        else
        {
            _1133 = g_ceil(_920 - 0.5);
        }
        float _944 = _367[2].y * 255.0;
        float _1134;
        if (_944 >= 0.0)
        {
            _1134 = g_floor(_944 + 0.5);
        }
        else
        {
            _1134 = g_ceil(_944 - 0.5);
        }
        float _968 = _367[3].y * 255.0;
        float _1135;
        if (_968 >= 0.0)
        {
            _1135 = g_floor(_968 + 0.5);
        }
        else
        {
            _1135 = g_ceil(_968 - 0.5);
        }
        _432._m0[_428 + 3u] = ((((uint)(g_clamp(_1132, 0.0, 255.0)) ^ 128u) | (((uint)(g_clamp(_1133, 0.0, 255.0)) ^ 128u) << (uint)(8))) | (((uint)(g_clamp(_1134, 0.0, 255.0)) ^ 128u) << (uint)(16))) | (((uint)(g_clamp(_1135, 0.0, 255.0)) ^ 128u) << (uint)(24));
        float _992 = _367[0].z * 255.0;
        float _1136;
        if (_992 >= 0.0)
        {
            _1136 = g_floor(_992 + 0.5);
        }
        else
        {
            _1136 = g_ceil(_992 - 0.5);
        }
        float _1016 = _367[1].z * 255.0;
        float _1137;
        if (_1016 >= 0.0)
        {
            _1137 = g_floor(_1016 + 0.5);
        }
        else
        {
            _1137 = g_ceil(_1016 - 0.5);
        }
        float _1040 = _367[2].z * 255.0;
        float _1138;
        if (_1040 >= 0.0)
        {
            _1138 = g_floor(_1040 + 0.5);
        }
        else
        {
            _1138 = g_ceil(_1040 - 0.5);
        }
        float _1064 = _367[3].z * 255.0;
        float _1139;
        if (_1064 >= 0.0)
        {
            _1139 = g_floor(_1064 + 0.5);
        }
        else
        {
            _1139 = g_ceil(_1064 - 0.5);
        }
        _432._m0[_428 + 4u] = ((((uint)(g_clamp(_1136, 0.0, 255.0)) ^ 128u) | (((uint)(g_clamp(_1137, 0.0, 255.0)) ^ 128u) << (uint)(8))) | (((uint)(g_clamp(_1138, 0.0, 255.0)) ^ 128u) << (uint)(16))) | (((uint)(g_clamp(_1139, 0.0, 255.0)) ^ 128u) << (uint)(24));
        _432._m0[_428 + 5u] = 0u;
        _432._m0[_428 + 6u] = 0u;
        _432._m0[_428 + 7u] = 0u;
        break;
    } while(false);
}
