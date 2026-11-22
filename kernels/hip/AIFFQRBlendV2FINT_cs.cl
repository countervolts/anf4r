// anf4r: OpenCL C port of Qualcomm's main_cs pass, translated from SPIRV-Cross GLSL by tools/glsl2cl.py.
// SPDX-License-Identifier: GPL-3.0-only
#define LX 16
#define LY 16
#define LZ 1
#include "anf_port.h"

#define AIFF_FULL_WIDTH ((int)SPECD(0, 0))
#define AIFF_FULL_HEIGHT ((int)SPECD(1, 0))
#define AIFF_LR_WIDTH ((int)SPECD(6, 0))
#define AIFF_LR_HEIGHT ((int)SPECD(7, 0))
#define AIFF_QR_FLOW_OUTPUT_WIDTH ((int)SPECD(16, 0))
#define AIFF_QR_FLOW_OUTPUT_HEIGHT ((int)SPECD(17, 0))
#define AIFF_QR_MASK_OUTPUT_WIDTH ((int)SPECD(20, 0))
#define AIFF_QR_MASK_OUTPUT_HEIGHT ((int)SPECD(21, 0))
#define _72 ((int)((AIFF_QR_FLOW_OUTPUT_WIDTH - 1)))
#define _73 ((int)((AIFF_QR_FLOW_OUTPUT_HEIGHT - 1)))
typedef struct {
    uint _m0[];
} type_ByteAddressBuffer_t;
#define bufFallbackStats (*(global type_ByteAddressBuffer_t*)SLOT(4))
typedef struct {
    float discontinuityThresholdSq;
} type_ConstantBuffer_QRBlendFINTPushConstants_t;
#define fintParams (*(constant const type_ConstantBuffer_QRBlendFINTPushConstants_t*)a->push)
#define texPriorFrame TEX(0)
#define texCurrFrame TEX(1)
#define texQRFlowUnpacked TEX(2)
#define texQRMaskUnpacked TEX(3)
#define samplerLinear SMP(128)
#define texOutput TEX(384)

ANF_KERNEL(anf_main_cs)
{
    uint _85 = (uint)(AIFF_LR_WIDTH) * ((uint)(AIFF_LR_HEIGHT) - 1u);
    do
    {
        int2 _93 = ivec2(((uint3)(GID_X, GID_Y, GID_Z)).xy);
        int _94 = _93.x;
        bool _100;
        if (_94 < AIFF_FULL_WIDTH)
        {
            _100 = _93.y >= AIFF_FULL_HEIGHT;
        }
        else
        {
            _100 = true;
        }
        if (_100)
        {
            break;
        }
        bool _111;
        if (bufFallbackStats._m0[0u] <= (8u * _85))
        {
            _111 = bufFallbackStats._m0[1u] > ((_85 * 20u) / 100u);
        }
        else
        {
            _111 = true;
        }
        if (_111)
        {
            imageStore(texOutput, ivec2(((uint3)(GID_X, GID_Y, GID_Z)).xy), vec4(g_clamp(texelFetch(texCurrFrame, ivec3(_94, _93.y, 0).xy, 0).xyz, vec3(0.0), vec3(1.0)), 1.0));
            break;
        }
        float _126 = (float)(AIFF_FULL_WIDTH);
        float _128 = (float)(AIFF_FULL_HEIGHT);
        float2 _130 = vec2(1.0 / _126, 1.0 / _128);
        float _131 = (float)(AIFF_QR_MASK_OUTPUT_WIDTH);
        float _133 = (float)(AIFF_QR_MASK_OUTPUT_HEIGHT);
        int2 _136 = ivec2(_72, _73);
        int2 _138 = g_clamp(_93 / ivec2(4), ivec2(0), _136);
        int2 _139 = _93 - ivec2(4) * (_93 / ivec2(4));
        float _141 = (float)(_139.y);
        float _144 = (float)(_139.x);
        float _146 = _141 * 0.125;
        float _147 = 1.0 - (_141 * 0.25);
        float _148 = _147 * 0.5;
        float _149 = _144 * 0.125;
        float _150 = 1.0 - (_144 * 0.25);
        float _151 = _150 * 0.5;
        float4 _157 = texelFetch(texQRFlowUnpacked, ivec3(_138, 0).xy, 0);
        int2 _163 = ivec3(g_clamp(_138 + ivec2(0, -1), ivec2(0), _136), 0).xy;
        float4 _165 = texelFetch(texQRFlowUnpacked, _163, 0);
        int2 _171 = ivec3(g_clamp(_138 + ivec2(0, 1), ivec2(0), _136), 0).xy;
        float4 _173 = texelFetch(texQRFlowUnpacked, _171, 0);
        int2 _179 = ivec3(g_clamp(_138 + ivec2(-1, 0), ivec2(0), _136), 0).xy;
        float4 _181 = texelFetch(texQRFlowUnpacked, _179, 0);
        int2 _187 = ivec3(g_clamp(_138 + ivec2(1, 0), ivec2(0), _136), 0).xy;
        float4 _189 = texelFetch(texQRFlowUnpacked, _187, 0);
        float2 _190 = _157.xy;
        float2 _191 = _157.zw;
        float2 _193 = _165.xy - _190;
        float2 _196 = _173.xy - _190;
        float2 _200 = _181.xy - _190;
        float2 _203 = _189.xy - _190;
        bool _209 = fintParams.discontinuityThresholdSq > 0.0;
        bool _213;
        if (_209)
        {
            _213 = spvNMax(spvNMax(g_dot(_193, _193), g_dot(_196, _196)), spvNMax(g_dot(_200, _200), g_dot(_203, _203))) < fintParams.discontinuityThresholdSq;
        }
        else
        {
            _213 = false;
        }
        float2 _215 = _165.zw - _191;
        float2 _218 = _173.zw - _191;
        float2 _222 = _181.zw - _191;
        float2 _225 = _189.zw - _191;
        bool _232;
        if (_209)
        {
            _232 = spvNMax(spvNMax(g_dot(_215, _215), g_dot(_218, _218)), spvNMax(g_dot(_222, _222), g_dot(_225, _225))) < fintParams.discontinuityThresholdSq;
        }
        else
        {
            _232 = false;
        }
        float2 _233 = vec2(_93);
        float2 _235 = (_233 + vec2(0.5)) * _130;
        float3 _556;
        float3 _557;
        if (_213 ? _232 : false)
        {
            _556 = textureLod0(texCurrFrame, samplerLinear, _235 + (_191 * _130), 0.0).xyz;
            _557 = textureLod0(texPriorFrame, samplerLinear, _235 + (_190 * _130), 0.0).xyz;
        }
        else
        {
            float4 _247 = texelFetch(texQRFlowUnpacked, ivec3(g_clamp(_138 + ivec2(1), ivec2(0), _136), 0).xy, 0);
            float _248 = _146 * _149;
            bool _249 = !_213;
            float3 _261;
            if (_249)
            {
                _261 = textureLod0(texPriorFrame, samplerLinear, _235 + (_247.xy * _130), 0.0).xyz * _248;
            }
            else
            {
                _261 = vec3(0.0);
            }
            bool _262 = !_232;
            float3 _274;
            if (_262)
            {
                _274 = textureLod0(texCurrFrame, samplerLinear, _235 + (_247.zw * _130), 0.0).xyz * _248;
            }
            else
            {
                _274 = vec3(0.0);
            }
            float4 _276 = texelFetch(texQRFlowUnpacked, _171, 0);
            float _277 = _141 * 0.0625;
            float3 _290;
            if (_249)
            {
                _290 = _261 + (textureLod0(texPriorFrame, samplerLinear, _235 + (_276.xy * _130), 0.0).xyz * _277);
            }
            else
            {
                _290 = _261;
            }
            float3 _303;
            if (_262)
            {
                _303 = _274 + (textureLod0(texCurrFrame, samplerLinear, _235 + (_276.zw * _130), 0.0).xyz * _277);
            }
            else
            {
                _303 = _274;
            }
            float4 _311 = texelFetch(texQRFlowUnpacked, ivec3(g_clamp(_138 + ivec2(-1, 1), ivec2(0), _136), 0).xy, 0);
            float _312 = _146 * _151;
            float3 _325;
            if (_249)
            {
                _325 = _290 + (textureLod0(texPriorFrame, samplerLinear, _235 + (_311.xy * _130), 0.0).xyz * _312);
            }
            else
            {
                _325 = _290;
            }
            float3 _338;
            if (_262)
            {
                _338 = _303 + (textureLod0(texCurrFrame, samplerLinear, _235 + (_311.zw * _130), 0.0).xyz * _312);
            }
            else
            {
                _338 = _303;
            }
            float4 _340 = texelFetch(texQRFlowUnpacked, _187, 0);
            float _341 = _144 * 0.0625;
            float3 _354;
            if (_249)
            {
                _354 = _325 + (textureLod0(texPriorFrame, samplerLinear, _235 + (_340.xy * _130), 0.0).xyz * _341);
            }
            else
            {
                _354 = _325;
            }
            float3 _367;
            if (_262)
            {
                _367 = _338 + (textureLod0(texCurrFrame, samplerLinear, _235 + (_340.zw * _130), 0.0).xyz * _341);
            }
            else
            {
                _367 = _338;
            }
            float3 _379;
            if (_249)
            {
                _379 = _354 + (textureLod0(texPriorFrame, samplerLinear, _235 + (_190 * _130), 0.0).xyz * 0.25);
            }
            else
            {
                _379 = _354;
            }
            float3 _391;
            if (_262)
            {
                _391 = _367 + (textureLod0(texCurrFrame, samplerLinear, _235 + (_191 * _130), 0.0).xyz * 0.25);
            }
            else
            {
                _391 = _367;
            }
            float4 _393 = texelFetch(texQRFlowUnpacked, _179, 0);
            float _394 = _150 * 0.25;
            float3 _407;
            if (_249)
            {
                _407 = _379 + (textureLod0(texPriorFrame, samplerLinear, _235 + (_393.xy * _130), 0.0).xyz * _394);
            }
            else
            {
                _407 = _379;
            }
            float3 _420;
            if (_262)
            {
                _420 = _391 + (textureLod0(texCurrFrame, samplerLinear, _235 + (_393.zw * _130), 0.0).xyz * _394);
            }
            else
            {
                _420 = _391;
            }
            float4 _428 = texelFetch(texQRFlowUnpacked, ivec3(g_clamp(_138 + ivec2(1, -1), ivec2(0), _136), 0).xy, 0);
            float _429 = _148 * _149;
            float3 _442;
            if (_249)
            {
                _442 = _407 + (textureLod0(texPriorFrame, samplerLinear, _235 + (_428.xy * _130), 0.0).xyz * _429);
            }
            else
            {
                _442 = _407;
            }
            float3 _455;
            if (_262)
            {
                _455 = _420 + (textureLod0(texCurrFrame, samplerLinear, _235 + (_428.zw * _130), 0.0).xyz * _429);
            }
            else
            {
                _455 = _420;
            }
            float4 _457 = texelFetch(texQRFlowUnpacked, _163, 0);
            float _458 = _147 * 0.25;
            float3 _471;
            if (_249)
            {
                _471 = _442 + (textureLod0(texPriorFrame, samplerLinear, _235 + (_457.xy * _130), 0.0).xyz * _458);
            }
            else
            {
                _471 = _442;
            }
            float3 _484;
            if (_262)
            {
                _484 = _455 + (textureLod0(texCurrFrame, samplerLinear, _235 + (_457.zw * _130), 0.0).xyz * _458);
            }
            else
            {
                _484 = _455;
            }
            float4 _492 = texelFetch(texQRFlowUnpacked, ivec3(g_clamp(_138 + ivec2(-1), ivec2(0), _136), 0).xy, 0);
            float _493 = _148 * _151;
            float3 _506;
            if (_249)
            {
                _506 = _471 + (textureLod0(texPriorFrame, samplerLinear, _235 + (_492.xy * _130), 0.0).xyz * _493);
            }
            else
            {
                _506 = _471;
            }
            float3 _519;
            if (_262)
            {
                _519 = _484 + (textureLod0(texCurrFrame, samplerLinear, _235 + (_492.zw * _130), 0.0).xyz * _493);
            }
            else
            {
                _519 = _484;
            }
            float3 _530;
            if (_213)
            {
                _530 = textureLod0(texPriorFrame, samplerLinear, _235 + (_190 * _130), 0.0).xyz;
            }
            else
            {
                _530 = _506;
            }
            float3 _541;
            if (_232)
            {
                _541 = textureLod0(texCurrFrame, samplerLinear, _235 + (_191 * _130), 0.0).xyz;
            }
            else
            {
                _541 = _519;
            }
            _556 = _541;
            _557 = _530;
        }
        float4 _564 = textureLod0(texQRMaskUnpacked, samplerLinear, ((_233 * 0.125) + vec2(0.0625)) * vec2(1.0 / _131, 1.0 / _133), 0.0);
        float _565 = _564.x;
        imageStore(texOutput, ivec2(((uint3)(GID_X, GID_Y, GID_Z)).xy), vec4(g_clamp((_557 * _565) + (_556 * (1.0 - _565)), vec3(0.0), vec3(1.0)), 1.0));
        break;
    } while(false);
}
