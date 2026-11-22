// anf4r: OpenCL C port of Qualcomm's main_cs pass, translated from SPIRV-Cross GLSL by tools/glsl2cl.py.
// SPDX-License-Identifier: GPL-3.0-only
#define LX 16
#define LY 16
#define LZ 1
#include "anf_port.h"

#define AIFF_LR_OUTPUT_WIDTH ((int)SPECD(8, 0))
#define AIFF_LR_OUTPUT_HEIGHT ((int)SPECD(9, 0))
#define AIFF_LR_MASK_OUTPUT_WIDTH ((int)SPECD(10, 0))
#define AIFF_LR_MASK_OUTPUT_HEIGHT ((int)SPECD(11, 0))
#define AIFF_QR_FLOW_OUTPUT_WIDTH_PADDED ((int)SPECD(14, 0))
#define AIFF_QR_FLOW_OUTPUT_WIDTH ((int)SPECD(16, 0))
#define AIFF_QR_FLOW_OUTPUT_HEIGHT ((int)SPECD(17, 0))
#define AIFF_QR_MASK_OUTPUT_WIDTH_PADDED ((int)SPECD(18, 0))
#define AIFF_QR_MASK_OUTPUT_WIDTH ((int)SPECD(20, 0))
#define AIFF_QR_MASK_OUTPUT_HEIGHT ((int)SPECD(21, 0))
#define QR_OUTPUT_FLOW_SCALE SPECDF(30, 0.0f)
#define QR_OUTPUT_FLOW_OFFSET_UINT8 SPECDF(31, 0.0f)
#define QR_OUTPUT_MASK_SCALE SPECDF(32, 0.0f)
#define QR_OUTPUT_MASK_OFFSET_UINT8 SPECDF(33, 0.0f)
#define _79 ((int)((AIFF_QR_FLOW_OUTPUT_WIDTH - 1)))
#define _80 ((int)((AIFF_QR_FLOW_OUTPUT_HEIGHT - 1)))
#define _81 ((int)((AIFF_QR_MASK_OUTPUT_WIDTH - 1)))
#define _82 ((int)((AIFF_QR_MASK_OUTPUT_HEIGHT - 1)))
typedef struct {
    uint _m0[];
} type_ByteAddressBuffer_t;
#define bufQRFlow (*(global type_ByteAddressBuffer_t*)SLOT(2))
typedef struct {
    uint _m0[];
} bufQRMask_t;
#define bufQRMask_1 (*(global bufQRMask_t*)SLOT(3))
typedef struct {
    int modeBias;
} type_ConstantBuffer_AIFFLRToQRPackPushConstants_t;
#define pc (*(constant const type_ConstantBuffer_AIFFLRToQRPackPushConstants_t*)a->push)
#define texLRFlowUnpacked TEX(0)
#define texLRMaskUnpacked TEX(1)
#define samplerLinear SMP(128)
#define texQRFlowUnpacked TEX(384)
#define texQRMaskUnpacked TEX(385)

ANF_KERNEL(anf_main_cs)
{
    int2 _93 = ivec2(((uint3)(GID_X, GID_Y, GID_Z)).xy);
    uint _97 = 128u - (uint)(pc.modeBias);
    int _98 = _93.x;
    bool _104;
    if (_98 < AIFF_QR_FLOW_OUTPUT_WIDTH)
    {
        _104 = _93.y < AIFF_QR_FLOW_OUTPUT_HEIGHT;
    }
    else
    {
        _104 = false;
    }
    if (_104)
    {
        int2 _108 = g_clamp(_93, ivec2(0), ivec2(_79, _80));
        uint _117 = ((((uint)(_108.y) * (uint)(AIFF_QR_FLOW_OUTPUT_WIDTH_PADDED)) + (uint)(_108.x)) * 4u) >> 2u;
        int4 _167 = ivec4(g_clamp((int)(g_floor(((((float)(((bufQRFlow._m0[_117] & 255u) + _97) & 255u) - QR_OUTPUT_FLOW_OFFSET_UINT8) * QR_OUTPUT_FLOW_SCALE) * 16384.0) + 0.5)), -32768, 32767), g_clamp((int)(g_floor(((((float)((((bufQRFlow._m0[_117] >> 8u) & 255u) + _97) & 255u) - QR_OUTPUT_FLOW_OFFSET_UINT8) * QR_OUTPUT_FLOW_SCALE) * 16384.0) + 0.5)), -32768, 32767), g_clamp((int)(g_floor(((((float)((((bufQRFlow._m0[_117] >> 16u) & 255u) + _97) & 255u) - QR_OUTPUT_FLOW_OFFSET_UINT8) * QR_OUTPUT_FLOW_SCALE) * 16384.0) + 0.5)), -32768, 32767), g_clamp((int)(g_floor(((((float)((((bufQRFlow._m0[_117] >> 24u) & 255u) + _97) & 255u) - QR_OUTPUT_FLOW_OFFSET_UINT8) * QR_OUTPUT_FLOW_SCALE) * 16384.0) + 0.5)), -32768, 32767));
        float _168 = (float)(AIFF_LR_OUTPUT_WIDTH);
        float _170 = (float)(AIFF_LR_OUTPUT_HEIGHT);
        float4 _180 = textureLod0(texLRFlowUnpacked, samplerLinear, ((vec2(_93) * 0.25) + vec2(0.125)) * vec2(1.0 / _168, 1.0 / _170), 0.0);
        imageStore(texQRFlowUnpacked, ivec2(((uint3)(GID_X, GID_Y, GID_Z)).xy), vec4(ivec4(g_clamp(_167.xy + g_clamp(ivec2(g_roundEven(_180.xy * 64.0)), ivec2(-32768), ivec2(32767)), ivec2(-32768), ivec2(32767)), g_clamp(_167.zw + g_clamp(ivec2(g_roundEven(_180.zw * 64.0)), ivec2(-32768), ivec2(32767)), ivec2(-32768), ivec2(32767)))) * 0.03125);
    }
    bool _212;
    if ((_98 & 1) == 0)
    {
        _212 = (_93.y & 1) == 0;
    }
    else
    {
        _212 = false;
    }
    if (_212)
    {
        int2 _215 = _93 / ivec2(2);
        bool _222;
        if (_215.x < AIFF_QR_MASK_OUTPUT_WIDTH)
        {
            _222 = _215.y < AIFF_QR_MASK_OUTPUT_HEIGHT;
        }
        else
        {
            _222 = false;
        }
        if (_222)
        {
            int2 _226 = g_clamp(_215, ivec2(0), ivec2(_81, _82));
            uint _233 = ((uint)(_226.y) * (uint)(AIFF_QR_MASK_OUTPUT_WIDTH_PADDED)) + (uint)(_226.x);
            float _253 = (float)(AIFF_LR_MASK_OUTPUT_WIDTH);
            float _255 = (float)(AIFF_LR_MASK_OUTPUT_HEIGHT);
            imageStore(texQRMaskUnpacked, ivec2(uvec2(_215)), vec4((float)(g_clamp(g_clamp((int)(g_roundEven((((float)((((bufQRMask_1._m0[((_233 / 4u) * 4u) >> 2u] >> (((_233 % 4u) * 8u) & 31u)) & 255u) + _97) & 255u) - QR_OUTPUT_MASK_OFFSET_UINT8) * QR_OUTPUT_MASK_SCALE) * 255.0)), -32768, 32767) + g_clamp((int)(g_roundEven(textureLod0(texLRMaskUnpacked, samplerLinear, ((vec2(_215) * 0.25) + vec2(0.125)) * vec2(1.0 / _253, 1.0 / _255), 0.0).x * 255.0)), 0, 255), 0, 255)) * 0.0039215688593685626983642578125));
        }
    }
}
