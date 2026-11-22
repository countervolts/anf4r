// anf4r: OpenCL C port of Qualcomm's main_cs pass, translated from SPIRV-Cross GLSL by tools/glsl2cl.py.
// SPDX-License-Identifier: GPL-3.0-only
#define LX 16
#define LY 16
#define LZ 1
#include "anf_port.h"

#define AIFF_FULL_WIDTH ((int)SPECD(0, 0))
#define AIFF_FULL_HEIGHT ((int)SPECD(1, 0))
#define AIFF_PREPROCESS_WIDTH ((int)SPECD(2, 0))
#define AIFF_PREPROCESS_HEIGHT ((int)SPECD(3, 0))
#define AIFF_LR_WIDTH ((int)SPECD(6, 0))
#define AIFF_LR_HEIGHT ((int)SPECD(7, 0))
#define AIFF_QR_WIDTH ((int)SPECD(12, 0))
#define AIFF_QR_HEIGHT ((int)SPECD(13, 0))
#define _104 ((int)((AIFF_PREPROCESS_WIDTH + 1)))
#define _105 ((int)((AIFF_PREPROCESS_HEIGHT + 1)))
#define _108 ((int)((AIFF_LR_WIDTH - 1)))
#define _109 ((int)((AIFF_LR_HEIGHT - 1)))
#define _110 ((int)((AIFF_PREPROCESS_WIDTH - 1)))
#define _111 ((int)((AIFF_PREPROCESS_HEIGHT - 1)))
#define _114 ((int)((AIFF_PREPROCESS_WIDTH + 4)))
#define _115 ((int)((AIFF_PREPROCESS_HEIGHT + 4)))
#define _117 ((int)((AIFF_PREPROCESS_HEIGHT - 1)))
#define _118 ((int)((AIFF_PREPROCESS_WIDTH - 1)))
#define _119 ((int)((AIFF_PREPROCESS_WIDTH - 1)))
#define _120 ((int)((AIFF_PREPROCESS_HEIGHT - 1)))
#define inputTexture TEX(0)
#define texDepthPyramidQR TEX(1)
#define samplerLinear SMP(128)
#define outputQRYUV TEX(384)
#define outputLRYUV TEX(385)
#define outputDepthLR TEX(386)

ANF_KERNEL(anf_main_cs)
{
    local float4 gs_L1[21][21];
    local float4 gs_L2[10][10];
    local float gs_depth_casc[7][7];
    float _134 = (float)(AIFF_FULL_WIDTH);
    float _135 = 1.0 / _134;
    float _136 = (float)(AIFF_FULL_HEIGHT);
    float _137 = 1.0 / _136;
    int _140 = (int)(((uint3)(LID_X, LID_Y, LID_Z)).x);
    int _143 = (int)(((uint3)(LID_X, LID_Y, LID_Z)).y);
    int _146 = (int)(((uint3)(GID_X, GID_Y, GID_Z)).x);
    int _149 = (int)(((uint3)(GID_X, GID_Y, GID_Z)).y);
    int _151 = (_143 * 16) + _140;
    int _154 = (int)(((uint3)(WG_X, WG_Y, WG_Z)).x);
    int _155 = _154 * 16;
    int _158 = (int)(((uint3)(WG_X, WG_Y, WG_Z)).y);
    int _159 = _158 * 16;
    float _164 = (float)(g_clamp(_146, 0, _119) * 2) * _135;
    float _167 = (float)(g_clamp(_149, 0, _120) * 2) * _137;
    float _175 = _164 + _135;
    float _181 = _167 + _137;
    float4 _193 = (((textureLod0(inputTexture, samplerLinear, vec2(_164, _167), 0.0) + textureLod0(inputTexture, samplerLinear, vec2(_175, _167), 0.0)) + textureLod0(inputTexture, samplerLinear, vec2(_164, _181), 0.0)) + textureLod0(inputTexture, samplerLinear, vec2(_175, _181), 0.0)) * 0.25;
    gs_L1[_143 + 3][_140 + 3] = _193;
    bool _201;
    if (_146 < AIFF_QR_WIDTH)
    {
        _201 = _149 < AIFF_QR_HEIGHT;
    }
    else
    {
        _201 = false;
    }
    if (_201)
    {
        int _204 = _104 / 2;
        bool _212;
        if (_146 < (_204 * 2))
        {
            int _209 = _105 / 2;
            _212 = _149 >= (_209 * 2);
        }
        else
        {
            _212 = true;
        }
        float4 _239;
        if (_212)
        {
            _239 = vec4(0.0, 0.0, 0.0, 1.0);
        }
        else
        {
            float3 _222 = g_roundEven(vec4(_193.xyz, 1.0).xyz * 255.0);
            _239 = vec4(g_clamp(g_floor(g_dot(_222, vec3(0.2125999927520751953125, 0.715200006961822509765625, 0.072200000286102294921875)) + 0.5), 0.0, 255.0) * 0.0039215688593685626983642578125, g_clamp(g_floor(g_dot(_222, vec3(-0.114600002765655517578125, -0.385399997234344482421875, 0.5)) + 128.0), 0.0, 255.0) * 0.0039215688593685626983642578125, g_clamp(g_floor(g_dot(_222, vec3(0.5, -0.4541999995708465576171875, -0.0458000004291534423828125)) + 128.0), 0.0, 255.0) * 0.0039215688593685626983642578125, 1.0);
        }
        imageStore(outputQRYUV, ivec2(uvec2(((uint3)(GID_X, GID_Y, GID_Z)).x, ((uint3)(GID_X, GID_Y, GID_Z)).y)), _239);
    }
    if (_151 < 21)
    {
        float _253 = (float)(g_clamp((_155 + _151) - 3, 0, _119) * 2) * _135;
        float _256 = (float)(g_clamp(_159 - 3, 0, _120) * 2) * _137;
        float _264 = _253 + _135;
        float _270 = _256 + _137;
        gs_L1[0][_151] = (((textureLod0(inputTexture, samplerLinear, vec2(_253, _256), 0.0) + textureLod0(inputTexture, samplerLinear, vec2(_264, _256), 0.0)) + textureLod0(inputTexture, samplerLinear, vec2(_253, _270), 0.0)) + textureLod0(inputTexture, samplerLinear, vec2(_264, _270), 0.0)) * 0.25;
    }
    else
    {
        if (_151 < 42)
        {
            int _288 = _151 - 21;
            float _296 = (float)(g_clamp((_155 + _288) - 3, 0, _119) * 2) * _135;
            float _299 = (float)(g_clamp(_159 - 2, 0, _120) * 2) * _137;
            float _307 = _296 + _135;
            float _313 = _299 + _137;
            gs_L1[1][_288] = (((textureLod0(inputTexture, samplerLinear, vec2(_296, _299), 0.0) + textureLod0(inputTexture, samplerLinear, vec2(_307, _299), 0.0)) + textureLod0(inputTexture, samplerLinear, vec2(_296, _313), 0.0)) + textureLod0(inputTexture, samplerLinear, vec2(_307, _313), 0.0)) * 0.25;
        }
        else
        {
            if (_151 < 63)
            {
                int _331 = _151 - 42;
                float _339 = (float)(g_clamp((_155 + _331) - 3, 0, _119) * 2) * _135;
                float _342 = (float)(g_clamp(_159 - 1, 0, _120) * 2) * _137;
                float _350 = _339 + _135;
                float _356 = _342 + _137;
                gs_L1[2][_331] = (((textureLod0(inputTexture, samplerLinear, vec2(_339, _342), 0.0) + textureLod0(inputTexture, samplerLinear, vec2(_350, _342), 0.0)) + textureLod0(inputTexture, samplerLinear, vec2(_339, _356), 0.0)) + textureLod0(inputTexture, samplerLinear, vec2(_350, _356), 0.0)) * 0.25;
            }
            else
            {
                if (_151 < 84)
                {
                    int _374 = _151 - 63;
                    float _382 = (float)(g_clamp((_155 + _374) - 3, 0, _119) * 2) * _135;
                    float _385 = (float)(g_clamp(_159 + 16, 0, _120) * 2) * _137;
                    float _393 = _382 + _135;
                    float _399 = _385 + _137;
                    gs_L1[19][_374] = (((textureLod0(inputTexture, samplerLinear, vec2(_382, _385), 0.0) + textureLod0(inputTexture, samplerLinear, vec2(_393, _385), 0.0)) + textureLod0(inputTexture, samplerLinear, vec2(_382, _399), 0.0)) + textureLod0(inputTexture, samplerLinear, vec2(_393, _399), 0.0)) * 0.25;
                }
                else
                {
                    if (_151 < 105)
                    {
                        int _417 = _151 - 84;
                        float _425 = (float)(g_clamp((_155 + _417) - 3, 0, _119) * 2) * _135;
                        float _428 = (float)(g_clamp(_159 + 17, 0, _120) * 2) * _137;
                        float _436 = _425 + _135;
                        float _442 = _428 + _137;
                        gs_L1[20][_417] = (((textureLod0(inputTexture, samplerLinear, vec2(_425, _428), 0.0) + textureLod0(inputTexture, samplerLinear, vec2(_436, _428), 0.0)) + textureLod0(inputTexture, samplerLinear, vec2(_425, _442), 0.0)) + textureLod0(inputTexture, samplerLinear, vec2(_436, _442), 0.0)) * 0.25;
                    }
                    else
                    {
                        if (_151 < 121)
                        {
                            int _460 = _151 + (-102);
                            float _468 = (float)(g_clamp(_155 - 3, 0, _119) * 2) * _135;
                            float _471 = (float)(g_clamp((_159 + _460) - 3, 0, _120) * 2) * _137;
                            float _479 = _468 + _135;
                            float _485 = _471 + _137;
                            gs_L1[_460][0] = (((textureLod0(inputTexture, samplerLinear, vec2(_468, _471), 0.0) + textureLod0(inputTexture, samplerLinear, vec2(_479, _471), 0.0)) + textureLod0(inputTexture, samplerLinear, vec2(_468, _485), 0.0)) + textureLod0(inputTexture, samplerLinear, vec2(_479, _485), 0.0)) * 0.25;
                        }
                        else
                        {
                            if (_151 < 137)
                            {
                                int _503 = _151 + (-118);
                                float _511 = (float)(g_clamp(_155 - 2, 0, _119) * 2) * _135;
                                float _514 = (float)(g_clamp((_159 + _503) - 3, 0, _120) * 2) * _137;
                                float _522 = _511 + _135;
                                float _528 = _514 + _137;
                                gs_L1[_503][1] = (((textureLod0(inputTexture, samplerLinear, vec2(_511, _514), 0.0) + textureLod0(inputTexture, samplerLinear, vec2(_522, _514), 0.0)) + textureLod0(inputTexture, samplerLinear, vec2(_511, _528), 0.0)) + textureLod0(inputTexture, samplerLinear, vec2(_522, _528), 0.0)) * 0.25;
                            }
                            else
                            {
                                if (_151 < 153)
                                {
                                    int _546 = _151 + (-134);
                                    float _554 = (float)(g_clamp(_155 - 1, 0, _119) * 2) * _135;
                                    float _557 = (float)(g_clamp((_159 + _546) - 3, 0, _120) * 2) * _137;
                                    float _565 = _554 + _135;
                                    float _571 = _557 + _137;
                                    gs_L1[_546][2] = (((textureLod0(inputTexture, samplerLinear, vec2(_554, _557), 0.0) + textureLod0(inputTexture, samplerLinear, vec2(_565, _557), 0.0)) + textureLod0(inputTexture, samplerLinear, vec2(_554, _571), 0.0)) + textureLod0(inputTexture, samplerLinear, vec2(_565, _571), 0.0)) * 0.25;
                                }
                                else
                                {
                                    if (_151 < 169)
                                    {
                                        int _589 = _151 + (-150);
                                        float _597 = (float)(g_clamp(_155 + 16, 0, _119) * 2) * _135;
                                        float _600 = (float)(g_clamp((_159 + _589) - 3, 0, _120) * 2) * _137;
                                        float _608 = _597 + _135;
                                        float _614 = _600 + _137;
                                        gs_L1[_589][19] = (((textureLod0(inputTexture, samplerLinear, vec2(_597, _600), 0.0) + textureLod0(inputTexture, samplerLinear, vec2(_608, _600), 0.0)) + textureLod0(inputTexture, samplerLinear, vec2(_597, _614), 0.0)) + textureLod0(inputTexture, samplerLinear, vec2(_608, _614), 0.0)) * 0.25;
                                    }
                                    else
                                    {
                                        if (_151 < 185)
                                        {
                                            int _631 = _151 + (-166);
                                            float _639 = (float)(g_clamp(_155 + 17, 0, _119) * 2) * _135;
                                            float _642 = (float)(g_clamp((_159 + _631) - 3, 0, _120) * 2) * _137;
                                            float _650 = _639 + _135;
                                            float _656 = _642 + _137;
                                            gs_L1[_631][20] = (((textureLod0(inputTexture, samplerLinear, vec2(_639, _642), 0.0) + textureLod0(inputTexture, samplerLinear, vec2(_650, _642), 0.0)) + textureLod0(inputTexture, samplerLinear, vec2(_639, _656), 0.0)) + textureLod0(inputTexture, samplerLinear, vec2(_650, _656), 0.0)) * 0.25;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    LDS_BARRIER();
    bool _674;
    if (_140 < 10)
    {
        _674 = _143 < 10;
    }
    else
    {
        _674 = false;
    }
    if (_674)
    {
        uint _677 = ((uint3)(LID_X, LID_Y, LID_Z)).y * 2u;
        uint _678 = ((uint3)(LID_X, LID_Y, LID_Z)).x * 2u;
        uint _683;
        if (((uint3)(WG_X, WG_Y, WG_Z)).y == 0u)
        {
            _683 = g_max(_677, 2u);
        }
        else
        {
            _683 = _677;
        }
        uint _688;
        if (((uint3)(WG_X, WG_Y, WG_Z)).x == 0u)
        {
            _688 = g_max(_678, 2u);
        }
        else
        {
            _688 = _678;
        }
        uint _691 = _688 + 1u;
        uint _694 = _688 + 2u;
        uint _697 = _683 + 1u;
        uint _704 = _683 + 2u;
        gs_L2[_143][_140] = ((((((gs_L1[_683][_688] + (gs_L1[_683][_691] * 2.0)) + gs_L1[_683][_694]) + (((gs_L1[_697][_688] + (gs_L1[_697][_691] * 2.0)) + gs_L1[_697][_694]) * 2.0)) + gs_L1[_704][_688]) + (gs_L1[_704][_691] * 2.0)) + gs_L1[_704][_694]) * 0.0625;
    }
    LDS_BARRIER();
    int2 _728 = ivec2(_154 * 4, _158 * 4) - ivec2(1);
    for (int _730 = _151; _730 < 49; _730 += 256)
    {
        int _736 = _730 - 7 * (_730 / 7);
        int _737 = _730 / 7;
        int2 _741 = g_clamp(_728 + ivec2(_736, _737), ivec2(0), ivec2(_108, _109));
        int _743 = _741.y * 4;
        float _775;
        if (_743 < AIFF_PREPROCESS_HEIGHT)
        {
            float _748;
            int _751;
            _748 = 0.0;
            _751 = 0;
            float _749;
            for (; _751 < 4; _748 = _749, _751++)
            {
                _749 = _748;
                for (int _759 = 0; _759 < 4; )
                {
                    float4 _772 = texelFetch(texDepthPyramidQR, ivec3(g_clamp((_741.x * 4) + _759, 0, _110), g_clamp(_743 + _751, 0, _111), 0).xy, 0);
                    float _773 = _772.x;
                    _749 = (_773 > _749) ? _773 : _749;
                    _759++;
                    continue;
                }
            }
            _775 = _748;
        }
        else
        {
            _775 = 0.0;
        }
        gs_depth_casc[_737][_736] = _775;
    }
    LDS_BARRIER();
    bool _781;
    if (_140 < 4)
    {
        _781 = _143 < 4;
    }
    else
    {
        _781 = false;
    }
    if (_781)
    {
        uint _785 = (((uint3)(WG_X, WG_Y, WG_Z)).x * 4u) + ((uint3)(LID_X, LID_Y, LID_Z)).x;
        uint _787 = (((uint3)(WG_X, WG_Y, WG_Z)).y * 4u) + ((uint3)(LID_X, LID_Y, LID_Z)).y;
        int _788 = (int)(_785);
        int _789 = (int)(_787);
        int2 _790 = ivec2(_788, _789);
        int _791 = _114 - 1;
        int _793 = _115 - 1;
        int2 _795 = ivec2(_791 / 4, _793 / 4);
        bool _800 = any(greaterThanEqual(_790, (((_795 + ivec2(1)) / ivec2(2)) * ivec2(2))));
        int2 _802 = g_min(_790, (_795 - ivec2(1)));
        int _803 = _143 * 2;
        int _804 = _140 * 2;
        bool _805 = !_800;
        int _816;
        int _817;
        if (_805)
        {
            _816 = _804 + ((_802.x - _788) * 2);
            _817 = _803 + ((_802.y - _789) * 2);
        }
        else
        {
            _816 = _804;
            _817 = _803;
        }
        int _818 = _117 / 2;
        int _821 = (_818 - (_158 * 8)) + 1;
        int _822 = _118 / 2;
        int _825 = (_822 - (_154 * 8)) + 1;
        int _827 = g_clamp(_817 + 1, 0, _821);
        int _829 = g_clamp(_817 + 2, 0, _821);
        int _831 = g_clamp(_816 + 1, 0, _825);
        int _833 = g_clamp(_816 + 2, 0, _825);
        float4 _891;
        if (_800)
        {
            _891 = vec4(0.0, 0.0, 0.0, 1.0);
        }
        else
        {
            float3 _874 = g_roundEven(vec4((((((((gs_L2[_817][_816] + (gs_L2[_817][_831] * 2.0)) + gs_L2[_817][_833]) + (((gs_L2[_827][_816] + (gs_L2[_827][_831] * 2.0)) + gs_L2[_827][_833]) * 2.0)) + gs_L2[_829][_816]) + (gs_L2[_829][_831] * 2.0)) + gs_L2[_829][_833]) * 0.0625).xyz, 1.0).xyz * 255.0);
            _891 = vec4(g_clamp(g_floor(g_dot(_874, vec3(0.2125999927520751953125, 0.715200006961822509765625, 0.072200000286102294921875)) + 0.5), 0.0, 255.0) * 0.0039215688593685626983642578125, g_clamp(g_floor(g_dot(_874, vec3(-0.114600002765655517578125, -0.385399997234344482421875, 0.5)) + 128.0), 0.0, 255.0) * 0.0039215688593685626983642578125, g_clamp(g_floor(g_dot(_874, vec3(0.5, -0.4541999995708465576171875, -0.0458000004291534423828125)) + 128.0), 0.0, 255.0) * 0.0039215688593685626983642578125, 1.0);
        }
        uint2 _892 = uvec2(_785, _787);
        imageStore(outputLRYUV, ivec2(_892), _891);
        int2 _896 = ivec2(_140 + 1, _143 + 1);
        int2 _901;
        if (_805)
        {
            _901 = _896 + (_802 - _790);
        }
        else
        {
            _901 = _896;
        }
        float _929;
        if (_805)
        {
            float _905;
            int _908;
            _905 = 0.0;
            _908 = 0;
            float _906;
            for (; _908 < 4; _905 = _906, _908++)
            {
                _906 = _905;
                for (int _916 = 0; _916 < 4; )
                {
                    int _922 = (_901.y - 1) + _908;
                    int _925 = (_901.x - 1) + _916;
                    _906 = (gs_depth_casc[_922][_925] > _906) ? gs_depth_casc[_922][_925] : _906;
                    _916++;
                    continue;
                }
            }
            _929 = _905;
        }
        else
        {
            _929 = 0.0;
        }
        imageStore(outputDepthLR, ivec2(_892), vec4(_929));
    }
}
