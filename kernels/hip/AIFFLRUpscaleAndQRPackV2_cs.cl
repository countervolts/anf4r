// anf4r: OpenCL C port of Qualcomm's main_cs pass, translated from SPIRV-Cross GLSL by tools/glsl2cl.py.
// SPDX-License-Identifier: GPL-3.0-only
#define LX 16
#define LY 16
#define LZ 1
#include "anf_port.h"

#define AIFF_PREPROCESS_WIDTH ((int)SPECD(2, 0))
#define AIFF_PREPROCESS_HEIGHT ((int)SPECD(3, 0))
#define AIFF_QR_WIDTH ((int)SPECD(12, 0))
#define AIFF_QR_HEIGHT ((int)SPECD(13, 0))
#define QR_INPUT_SCALE SPECDF(28, 0.0f)
#define QR_INPUT_OFFSET SPECDF(29, 0.0f)
typedef struct {
    uint _m0[];
} type_RWByteAddressBuffer_t;
#define bufNetworkInputQR (*(global type_RWByteAddressBuffer_t*)SLOT(384))
typedef struct {
    int modeBias;
} type_ConstantBuffer_AIFFLRToQRPackPushConstants_t;
#define pc (*(constant const type_ConstantBuffer_AIFFLRToQRPackPushConstants_t*)a->push)
#define texPrevYUV TEX(0)
#define texCurrYUV TEX(1)
#define texHoleMask TEX(2)
#define texMVFwd TEX(3)
#define texMVBwd TEX(4)
#define texDepth TEX(5)
#define texLRFlowUnpacked TEX(6)
#define texLRMaskUnpacked TEX(7)
#define texDepthPyramidQRPrev TEX(8)
#define texDepthPyramidQRCurr TEX(9)
#define samplerLinear SMP(128)

ANF_KERNEL(anf_main_cs)
{
    float _91 = (float)(AIFF_QR_WIDTH);
    float _92 = 0.0078125 / _91;
    float _93 = (float)(AIFF_QR_HEIGHT);
    float _94 = 0.0078125 / _93;
    float _95 = 0.015625 / _91;
    float _96 = 0.015625 / _93;
    float2 _99 = vec2(1.0 / _91, 1.0 / _93);
    int2 _102 = ivec2(((uint3)(GID_X, GID_Y, GID_Z)).xy);
    int2 _103 = ivec2(AIFF_PREPROCESS_WIDTH, AIFF_PREPROCESS_HEIGHT);
    bool _110 = any(greaterThanEqual(_102, (((_103 + ivec2(1)) / ivec2(2)) * ivec2(2))));
    int2 _112 = g_min(_102, (_103 - ivec2(1)));
    float2 _117 = (vec2(_102) + vec2(0.5)) * _99;
    float4 _119 = textureLod0(texLRFlowUnpacked, samplerLinear, _117, 0.0);
    int2 _124 = g_clamp(ivec2(g_roundEven(_119.xy * 64.0)), ivec2(-32768), ivec2(32767));
    int2 _129 = g_clamp(ivec2(g_roundEven(_119.zw * 64.0)), ivec2(-32768), ivec2(32767));
    bool _134;
    if (!all(lessThan(_102, _103)))
    {
        _134 = !_110;
    }
    else
    {
        _134 = false;
    }
    int2 _154;
    int2 _155;
    if (_134)
    {
        float4 _143 = textureLod0(texLRFlowUnpacked, samplerLinear, (vec2(_112) + vec2(0.5)) * _99, 0.0);
        _154 = g_clamp(ivec2(g_roundEven(_143.zw * 64.0)), ivec2(-32768), ivec2(32767));
        _155 = g_clamp(ivec2(g_roundEven(_143.xy * 64.0)), ivec2(-32768), ivec2(32767));
    }
    else
    {
        _154 = _129;
        _155 = _124;
    }
    int2 _176;
    if (_110)
    {
        _176 = ivec2(0);
    }
    else
    {
        _176 = texelFetchI(texMVFwd, ivec3(_112, 0).xy, 0).xy;
    }
    int2 _187;
    if (_110)
    {
        _187 = ivec2(0);
    }
    else
    {
        _187 = texelFetchI(texMVBwd, ivec3(_112, 0).xy, 0).xy;
    }
    float2 _188 = vec2(_91, _93);
    float _189 = (float)(AIFF_PREPROCESS_WIDTH);
    float _190 = (float)(AIFF_PREPROCESS_HEIGHT);
    float2 _193 = (vec2(_189, _190) - vec2(0.5)) / _188;
    float2 _196 = (vec2(_112) + vec2(0.5)) / _188;
    float _210;
    if (_110)
    {
        _210 = 0.0;
    }
    else
    {
        _210 = textureLod0(texPrevYUV, samplerLinear, spvNMin(_196 + (vec2(_176) * vec2(_92, _94)), _193), 0.0).x;
    }
    float _224;
    if (_110)
    {
        _224 = 0.0;
    }
    else
    {
        _224 = textureLod0(texCurrYUV, samplerLinear, spvNMin(_196 + (vec2(_187) * vec2(_92, _94)), _193), 0.0).x;
    }
    float _238;
    if (_110)
    {
        _238 = 0.0;
    }
    else
    {
        _238 = textureLod0(texPrevYUV, samplerLinear, spvNMin(_196 + (vec2(_155) * vec2(_95, _96)), _193), 0.0).x;
    }
    float _252;
    if (_110)
    {
        _252 = 0.0;
    }
    else
    {
        _252 = textureLod0(texCurrYUV, samplerLinear, spvNMin(_196 + (vec2(_154) * vec2(_95, _96)), _193), 0.0).x;
    }
    float _266;
    if (_110)
    {
        _266 = 0.0;
    }
    else
    {
        _266 = textureLod0(texDepthPyramidQRPrev, samplerLinear, spvNMin(_196 + (vec2(_176) * vec2(_92, _94)), _193), 0.0).x;
    }
    float _280;
    if (_110)
    {
        _280 = 0.0;
    }
    else
    {
        _280 = textureLod0(texDepthPyramidQRCurr, samplerLinear, spvNMin(_196 + (vec2(_187) * vec2(_92, _94)), _193), 0.0).x;
    }
    float _294;
    if (_110)
    {
        _294 = 0.0;
    }
    else
    {
        _294 = textureLod0(texDepthPyramidQRPrev, samplerLinear, spvNMin(_196 + (vec2(_155) * vec2(_95, _96)), _193), 0.0).x;
    }
    float _308;
    if (_110)
    {
        _308 = 0.0;
    }
    else
    {
        _308 = textureLod0(texDepthPyramidQRCurr, samplerLinear, spvNMin(_196 + (vec2(_154) * vec2(_95, _96)), _193), 0.0).x;
    }
    uint _316;
    if (_110)
    {
        _316 = 0u;
    }
    else
    {
        _316 = texelFetchU(texHoleMask, ivec2(uvec2(_112)), (int)(0u)).x;
    }
    uint _324;
    if (_110)
    {
        _324 = 0u;
    }
    else
    {
        _324 = texelFetchU(texDepth, ivec2(uvec2(_112)), (int)(0u)).x;
    }
    float _331 = (float)(pc.modeBias);
    int _336 = (-128) + pc.modeBias;
    int _337 = 127 + pc.modeBias;
    uint _557 = (((uint)(_102.y) * (uint)(AIFF_QR_WIDTH)) + (uint)(_102.x)) * 32u;
    uint _586 = _557 >> 2u;
    bufNetworkInputQR._m0[_586] = ((((uint)(g_clamp((int)(g_floor((((((_210 * 2.0) - 1.0) / QR_INPUT_SCALE) + QR_INPUT_OFFSET) + _331) + 0.5)), _336, _337)) & 255u) | (((uint)(g_clamp((int)(g_floor((((((_266 * 2.0) - 1.0) / QR_INPUT_SCALE) + QR_INPUT_OFFSET) + _331) + 0.5)), _336, _337)) & 255u) << 8u)) | (((uint)(g_clamp((int)(g_floor((((((_224 * 2.0) - 1.0) / QR_INPUT_SCALE) + QR_INPUT_OFFSET) + _331) + 0.5)), _336, _337)) & 255u) << 16u)) | (((uint)(g_clamp((int)(g_floor((((((_280 * 2.0) - 1.0) / QR_INPUT_SCALE) + QR_INPUT_OFFSET) + _331) + 0.5)), _336, _337)) & 255u) << 24u);
    bufNetworkInputQR._m0[_586 + 1u] = ((((uint)(g_clamp((int)(g_floor(((((((float)(_316) * 0.007843137718737125396728515625) - 1.0) / QR_INPUT_SCALE) + QR_INPUT_OFFSET) + _331) + 0.5)), _336, _337)) & 255u) | (((uint)(g_clamp((int)(g_floor((((((_238 * 2.0) - 1.0) / QR_INPUT_SCALE) + QR_INPUT_OFFSET) + _331) + 0.5)), _336, _337)) & 255u) << 8u)) | (((uint)(g_clamp((int)(g_floor((((((_294 * 2.0) - 1.0) / QR_INPUT_SCALE) + QR_INPUT_OFFSET) + _331) + 0.5)), _336, _337)) & 255u) << 16u)) | (((uint)(g_clamp((int)(g_floor((((((_252 * 2.0) - 1.0) / QR_INPUT_SCALE) + QR_INPUT_OFFSET) + _331) + 0.5)), _336, _337)) & 255u) << 24u);
    bufNetworkInputQR._m0[_586 + 2u] = ((((uint)(g_clamp((int)(g_floor((((((_308 * 2.0) - 1.0) / QR_INPUT_SCALE) + QR_INPUT_OFFSET) + _331) + 0.5)), _336, _337)) & 255u) | (((uint)(g_clamp((int)(g_floor((((((float)(_124.x) * 6.103515625e-05) / QR_INPUT_SCALE) + QR_INPUT_OFFSET) + _331) + 0.5)), _336, _337)) & 255u) << 8u)) | (((uint)(g_clamp((int)(g_floor((((((float)(_124.y) * 6.103515625e-05) / QR_INPUT_SCALE) + QR_INPUT_OFFSET) + _331) + 0.5)), _336, _337)) & 255u) << 16u)) | (((uint)(g_clamp((int)(g_floor((((((float)(_129.x) * 6.103515625e-05) / QR_INPUT_SCALE) + QR_INPUT_OFFSET) + _331) + 0.5)), _336, _337)) & 255u) << 24u);
    bufNetworkInputQR._m0[_586 + 3u] = ((((uint)(g_clamp((int)(g_floor((((((float)(_129.y) * 6.103515625e-05) / QR_INPUT_SCALE) + QR_INPUT_OFFSET) + _331) + 0.5)), _336, _337)) & 255u) | (((uint)(g_clamp((int)(g_floor((((((float)(_176.x) * 3.0517578125e-05) / QR_INPUT_SCALE) + QR_INPUT_OFFSET) + _331) + 0.5)), _336, _337)) & 255u) << 8u)) | (((uint)(g_clamp((int)(g_floor((((((float)(_176.y) * 3.0517578125e-05) / QR_INPUT_SCALE) + QR_INPUT_OFFSET) + _331) + 0.5)), _336, _337)) & 255u) << 16u)) | (((uint)(g_clamp((int)(g_floor((((((float)(_187.x) * 3.0517578125e-05) / QR_INPUT_SCALE) + QR_INPUT_OFFSET) + _331) + 0.5)), _336, _337)) & 255u) << 24u);
    uint _595 = (_557 + 16u) >> 2u;
    bufNetworkInputQR._m0[_595] = (((uint)(g_clamp((int)(g_floor((((((float)(_187.y) * 3.0517578125e-05) / QR_INPUT_SCALE) + QR_INPUT_OFFSET) + _331) + 0.5)), _336, _337)) & 255u) | (((uint)(g_clamp((int)(g_floor(((((((float)((uint)(g_clamp((int)(g_roundEven(textureLod0(texLRMaskUnpacked, samplerLinear, _117, 0.0).x * 255.0)), 0, 255))) * 0.007843137718737125396728515625) - 1.0) / QR_INPUT_SCALE) + QR_INPUT_OFFSET) + _331) + 0.5)), _336, _337)) & 255u) << 8u)) | (((uint)(g_clamp((int)(g_floor(((((((float)(_324) * 0.007843137718737125396728515625) - 1.0) / QR_INPUT_SCALE) + QR_INPUT_OFFSET) + _331) + 0.5)), _336, _337)) & 255u) << 16u);
    bufNetworkInputQR._m0[_595 + 1u] = 0u;
    bufNetworkInputQR._m0[_595 + 2u] = 0u;
    bufNetworkInputQR._m0[_595 + 3u] = 0u;
}
