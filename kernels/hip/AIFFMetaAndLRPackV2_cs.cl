// anf4r: OpenCL C port of Qualcomm's main_cs pass, translated from SPIRV-Cross GLSL by tools/glsl2cl.py.
// SPDX-License-Identifier: GPL-3.0-only
#define LX 16
#define LY 16
#define LZ 1
#include "anf_port.h"

#define AIFF_PREPROCESS_WIDTH ((int)SPECD(2, 0))
#define AIFF_PREPROCESS_HEIGHT ((int)SPECD(3, 0))
#define AIFF_LR_WIDTH ((int)SPECD(6, 0))
#define AIFF_LR_HEIGHT ((int)SPECD(7, 0))
#define LR_INPUT_SCALE SPECDF(22, 0.0f)
#define LR_INPUT_OFFSET SPECDF(23, 0.0f)
#define _93 ((int)((AIFF_PREPROCESS_WIDTH - 1)))
#define _94 ((int)((AIFF_PREPROCESS_HEIGHT - 1)))
#define _97 ((int)((AIFF_LR_WIDTH - 1)))
#define _98 ((int)((AIFF_LR_HEIGHT - 1)))
#define _100 ((int)((AIFF_PREPROCESS_WIDTH + 4)))
#define _101 ((int)((AIFF_PREPROCESS_HEIGHT + 4)))
typedef struct {
    uint _m0[];
} type_RWByteAddressBuffer_t;
#define bufNetworkInputLR (*(global type_RWByteAddressBuffer_t*)SLOT(384))
typedef struct {
    uint _m0[];
} bufFallbackStats_t;
#define bufFallbackStats_1 (*(global bufFallbackStats_t*)SLOT(385))
typedef struct {
    int modeBias;
} type_ConstantBuffer_AIFFLRToQRPackPushConstants_t;
#define pc (*(constant const type_ConstantBuffer_AIFFLRToQRPackPushConstants_t*)a->push)
#define texSrcDepth TEX(0)
#define texSrcMask TEX(1)
#define texSrcMVFwd TEX(2)
#define texSrcMVBwd TEX(3)
#define texLRPrevYUV TEX(4)
#define texLRCurrYUV TEX(5)
#define texDepthLRPrev TEX(6)
#define texDepthLRCurr TEX(7)
#define samplerLinear SMP(128)

ANF_KERNEL(anf_main_cs)
{
    local uint gs_casc_d[19][19];
    local uint gs_casc_src[19][19];
    local uint gs_fallback_luma_sum;
    local uint gs_fallback_oos_count;
    float _112 = (float)(AIFF_LR_WIDTH);
    float _113 = 0.0078125 / _112;
    float _114 = (float)(AIFF_LR_HEIGHT);
    float _115 = 0.0078125 / _114;
    do
    {
        bool _121 = LIDX == 0u;
        if (_121)
        {
            gs_fallback_luma_sum = 0u;
            gs_fallback_oos_count = 0u;
        }
        LDS_BARRIER();
        int2 _133 = ivec2((int)(((uint3)(WG_X, WG_Y, WG_Z)).x * 16u), (int)(((uint3)(WG_X, WG_Y, WG_Z)).y * 16u)) - ivec2(1);
        for (uint _135 = LIDX; _135 < 361u; _135 += 256u)
        {
            int _142 = (int)(_135 % 19u);
            int _144 = (int)(_135 / 19u);
            int2 _148 = g_clamp(_133 + ivec2(_142, _144), ivec2(0), ivec2(_97, _98));
            int _150 = _148.x * 4;
            int _152 = _148.y * 4;
            int2 _153 = ivec2(_150, _152);
            int2 _187;
            uint _188;
            if (_152 < AIFF_PREPROCESS_HEIGHT)
            {
                int2 _158;
                uint _161;
                int _163;
                _158 = _153;
                _161 = 0u;
                _163 = 0;
                int2 _159;
                uint _162;
                for (; _163 < 4; _158 = _159, _161 = _162, _163++)
                {
                    _159 = _158;
                    _162 = _161;
                    for (int _172 = 0; _172 < 4; )
                    {
                        int2 _180 = ivec2(g_clamp(_150 + _172, 0, _93), g_clamp(_152 + _163, 0, _94));
                        uint4 _183 = texelFetchU(texSrcDepth, ivec2(uvec2(_180)), (int)(0u));
                        uint _184 = _183.x;
                        bool _185 = _184 > _162;
                        _159 = g_mix(_159, _180, bvec2(_185));
                        _162 = _185 ? _184 : _162;
                        _172++;
                        continue;
                    }
                }
                _187 = _158;
                _188 = _161;
            }
            else
            {
                _187 = _153;
                _188 = 0u;
            }
            gs_casc_d[_144][_142] = _188;
            gs_casc_src[_144][_142] = (uint)(_187.x) | ((uint)(_187.y) << 16u);
        }
        LDS_BARRIER();
        int2 _198 = ivec2(((uint3)(GID_X, GID_Y, GID_Z)).xy);
        int _199 = _198.x;
        bool _205;
        if (_199 < AIFF_LR_WIDTH)
        {
            _205 = _198.y >= AIFF_LR_HEIGHT;
        }
        else
        {
            _205 = true;
        }
        if (_205)
        {
            break;
        }
        int _208 = _100 - 1;
        int _209 = _208 / 4;
        int _210 = _101 - 1;
        int _211 = _210 / 4;
        int2 _212 = ivec2(_209, _211);
        bool _217 = all(lessThan(_198, _212));
        bool _219 = any(greaterThanEqual(_198, (((_212 + ivec2(1)) / ivec2(2)) * ivec2(2))));
        int2 _221 = g_min(_198, (_212 - ivec2(1)));
        int2 _224 = ivec2(((uint3)(LID_X, LID_Y, LID_Z)).xy) + ivec2(1);
        bool _229;
        if (!_217)
        {
            _229 = !_219;
        }
        else
        {
            _229 = false;
        }
        int2 _234;
        if (_229)
        {
            _234 = _224 + (_221 - _198);
        }
        else
        {
            _234 = _224;
        }
        uint _269;
        int _270;
        int _271;
        if (!_219)
        {
            uint _239;
            int _242;
            int _244;
            int _246;
            _239 = 0u;
            _242 = 0;
            _244 = 0;
            _246 = 0;
            uint _240;
            int _243;
            int _245;
            for (; _246 < 4; _239 = _240, _242 = _243, _244 = _245, _246++)
            {
                _240 = _239;
                _243 = _242;
                _245 = _244;
                for (int _254 = 0; _254 < 4; )
                {
                    int _262 = (_234.y - 1) + _246;
                    int _265 = (_234.x - 1) + _254;
                    bool _268 = gs_casc_d[_262][_265] > _240;
                    int _256 = _268 ? _254 : _243;
                    _240 = _268 ? gs_casc_d[_262][_265] : _240;
                    _254++;
                    _243 = _256;
                    _245 = _268 ? _246 : _245;
                    continue;
                }
            }
            _269 = _239;
            _270 = _242;
            _271 = _244;
        }
        else
        {
            _269 = 0u;
            _270 = 0;
            _271 = 0;
        }
        int _274 = (_234.y - 1) + _271;
        int _277 = (_234.x - 1) + _270;
        int2 _284 = ivec2((int)(gs_casc_src[_274][_277] & 65535u), (int)(gs_casc_src[_274][_277] >> 16u));
        int _285 = _221.x;
        uint _300;
        if (_219)
        {
            _300 = 0u;
        }
        else
        {
            _300 = texelFetchU(texSrcMask, ivec2(uvec2(ivec2(g_min((_285 * 4), _93), g_min((_221.y * 4), _94)))), (int)(0u)).x;
        }
        int2 _310;
        if (_219)
        {
            _310 = ivec2(0);
        }
        else
        {
            _310 = (texelFetchI(texSrcMVFwd, ivec2(uvec2(_284)), (int)(0u)).xy + ivec2(3)) >> ivec2(2);
        }
        int2 _320;
        if (_219)
        {
            _320 = ivec2(0);
        }
        else
        {
            _320 = (texelFetchI(texSrcMVBwd, ivec2(uvec2(_284)), (int)(0u)).xy + ivec2(3)) >> ivec2(2);
        }
        float _329;
        if (_219)
        {
            _329 = 0.0;
        }
        else
        {
            _329 = texelFetch(texDepthLRCurr, ivec3(_285, _221.y, 0).xy, 0).x;
        }
        float _338;
        if (_219)
        {
            _338 = 0.0;
        }
        else
        {
            _338 = texelFetch(texDepthLRPrev, ivec3(_285, _221.y, 0).xy, 0).x;
        }
        float3 _347;
        if (_219)
        {
            _347 = vec3(0.0);
        }
        else
        {
            _347 = texelFetch(texLRPrevYUV, ivec2(uvec2(_221)), (int)(0u)).xyz * 255.0;
        }
        float3 _356;
        if (_219)
        {
            _356 = vec3(0.0);
        }
        else
        {
            _356 = texelFetch(texLRCurrYUV, ivec2(uvec2(_221)), (int)(0u)).xyz * 255.0;
        }
        float2 _357 = vec2(_112, _114);
        float2 _362 = (vec2((float)(_209), (float)(_211)) - vec2(0.5)) / _357;
        float2 _365 = (vec2(_221) + vec2(0.5)) / _357;
        float _379;
        if (_219)
        {
            _379 = 0.0;
        }
        else
        {
            _379 = textureLod0(texLRPrevYUV, samplerLinear, spvNMin(_365 + (vec2(_310) * vec2(_113, _115)), _362), 0.0).x;
        }
        float _393;
        if (_219)
        {
            _393 = 0.0;
        }
        else
        {
            _393 = textureLod0(texLRCurrYUV, samplerLinear, spvNMin(_365 + (vec2(_320) * vec2(_113, _115)), _362), 0.0).x;
        }
        float _407;
        if (_219)
        {
            _407 = 0.0;
        }
        else
        {
            _407 = textureLod0(texDepthLRPrev, samplerLinear, spvNMin(_365 + (vec2(_310) * vec2(_113, _115)), _362), 0.0).x;
        }
        float _421;
        if (_219)
        {
            _421 = 0.0;
        }
        else
        {
            _421 = textureLod0(texDepthLRCurr, samplerLinear, spvNMin(_365 + (vec2(_320) * vec2(_113, _115)), _362), 0.0).x;
        }
        if (_217)
        {
            uint _433;
            if (_300 > 0u)
            {
                _433 = (uint)(g_roundEven(g_abs(_379 - _393) * 255.0));
            }
            else
            {
                _433 = 0u;
            }
            uint _445 = atomicAdd(&(gs_fallback_luma_sum), _433);
            uint _446 = atomicAdd(&(gs_fallback_oos_count), (uint)((uint)(g_abs(_320.x)) > 4096u) + (uint)((uint)(g_abs(_320.y)) > 4096u));
        }
        float _454 = (float)(pc.modeBias);
        int _459 = (-128) + pc.modeBias;
        int _460 = 127 + pc.modeBias;
        uint _668 = (((uint)(_198.y) * (uint)(AIFF_LR_WIDTH)) + (uint)(_199)) * 32u;
        uint _695 = _668 >> 2u;
        bufNetworkInputLR._m0[_695] = ((((uint)(g_clamp((int)(g_floor((((((_347.x * 0.007843137718737125396728515625) - 1.0) / LR_INPUT_SCALE) + LR_INPUT_OFFSET) + _454) + 0.5)), _459, _460)) & 255u) | (((uint)(g_clamp((int)(g_floor((((((_347.y * 0.007843137718737125396728515625) - 1.0) / LR_INPUT_SCALE) + LR_INPUT_OFFSET) + _454) + 0.5)), _459, _460)) & 255u) << 8u)) | (((uint)(g_clamp((int)(g_floor((((((_347.z * 0.007843137718737125396728515625) - 1.0) / LR_INPUT_SCALE) + LR_INPUT_OFFSET) + _454) + 0.5)), _459, _460)) & 255u) << 16u)) | (((uint)(g_clamp((int)(g_floor((((((_338 * 2.0) - 1.0) / LR_INPUT_SCALE) + LR_INPUT_OFFSET) + _454) + 0.5)), _459, _460)) & 255u) << 24u);
        bufNetworkInputLR._m0[_695 + 1u] = ((((uint)(g_clamp((int)(g_floor((((((_356.x * 0.007843137718737125396728515625) - 1.0) / LR_INPUT_SCALE) + LR_INPUT_OFFSET) + _454) + 0.5)), _459, _460)) & 255u) | (((uint)(g_clamp((int)(g_floor((((((_356.y * 0.007843137718737125396728515625) - 1.0) / LR_INPUT_SCALE) + LR_INPUT_OFFSET) + _454) + 0.5)), _459, _460)) & 255u) << 8u)) | (((uint)(g_clamp((int)(g_floor((((((_356.z * 0.007843137718737125396728515625) - 1.0) / LR_INPUT_SCALE) + LR_INPUT_OFFSET) + _454) + 0.5)), _459, _460)) & 255u) << 16u)) | (((uint)(g_clamp((int)(g_floor((((((_329 * 2.0) - 1.0) / LR_INPUT_SCALE) + LR_INPUT_OFFSET) + _454) + 0.5)), _459, _460)) & 255u) << 24u);
        bufNetworkInputLR._m0[_695 + 2u] = ((((uint)(g_clamp((int)(g_floor((((((_379 * 2.0) - 1.0) / LR_INPUT_SCALE) + LR_INPUT_OFFSET) + _454) + 0.5)), _459, _460)) & 255u) | (((uint)(g_clamp((int)(g_floor((((((_407 * 2.0) - 1.0) / LR_INPUT_SCALE) + LR_INPUT_OFFSET) + _454) + 0.5)), _459, _460)) & 255u) << 8u)) | (((uint)(g_clamp((int)(g_floor((((((_393 * 2.0) - 1.0) / LR_INPUT_SCALE) + LR_INPUT_OFFSET) + _454) + 0.5)), _459, _460)) & 255u) << 16u)) | (((uint)(g_clamp((int)(g_floor((((((_421 * 2.0) - 1.0) / LR_INPUT_SCALE) + LR_INPUT_OFFSET) + _454) + 0.5)), _459, _460)) & 255u) << 24u);
        bufNetworkInputLR._m0[_695 + 3u] = ((((uint)(g_clamp((int)(g_floor(((((((float)(_300) * 0.007843137718737125396728515625) - 1.0) / LR_INPUT_SCALE) + LR_INPUT_OFFSET) + _454) + 0.5)), _459, _460)) & 255u) | (((uint)(g_clamp((int)(g_floor((((((float)(_310.x) * 0.0001220703125) / LR_INPUT_SCALE) + LR_INPUT_OFFSET) + _454) + 0.5)), _459, _460)) & 255u) << 8u)) | (((uint)(g_clamp((int)(g_floor((((((float)(_310.y) * 0.0001220703125) / LR_INPUT_SCALE) + LR_INPUT_OFFSET) + _454) + 0.5)), _459, _460)) & 255u) << 16u)) | (((uint)(g_clamp((int)(g_floor((((((float)(_320.x) * 0.0001220703125) / LR_INPUT_SCALE) + LR_INPUT_OFFSET) + _454) + 0.5)), _459, _460)) & 255u) << 24u);
        uint _704 = (_668 + 16u) >> 2u;
        bufNetworkInputLR._m0[_704] = ((uint)(g_clamp((int)(g_floor((((((float)(_320.y) * 0.0001220703125) / LR_INPUT_SCALE) + LR_INPUT_OFFSET) + _454) + 0.5)), _459, _460)) & 255u) | (((uint)(g_clamp((int)(g_floor(((((((float)(_219 ? 0u : _269) * 0.007843137718737125396728515625) - 1.0) / LR_INPUT_SCALE) + LR_INPUT_OFFSET) + _454) + 0.5)), _459, _460)) & 255u) << 8u);
        bufNetworkInputLR._m0[_704 + 1u] = 0u;
        bufNetworkInputLR._m0[_704 + 2u] = 0u;
        bufNetworkInputLR._m0[_704 + 3u] = 0u;
        LDS_BARRIER();
        if (_121)
        {
            uint _716 = atomicAdd(&(bufFallbackStats_1._m0[0u]), gs_fallback_luma_sum);
            uint _719 = atomicAdd(&(bufFallbackStats_1._m0[1u]), gs_fallback_oos_count);
        }
        break;
    } while(false);
}
