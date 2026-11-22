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
#define _50 ((int)((AIFF_LR_OUTPUT_WIDTH - 1)))
#define _51 ((int)((AIFF_LR_OUTPUT_HEIGHT - 1)))
#define _54 ((int)((AIFF_LR_MASK_OUTPUT_WIDTH - 1)))
#define _55 ((int)((AIFF_LR_MASK_OUTPUT_HEIGHT - 1)))
typedef struct {
    uint _m0[];
} type_ByteAddressBuffer_t;
#define bufFlowLR (*(global type_ByteAddressBuffer_t*)SLOT(0))
typedef struct {
    uint _m0[];
} bufMaskLR_t;
#define bufMaskLR_1 (*(global bufMaskLR_t*)SLOT(1))
#define texLRFlowUnpacked TEX(384)
#define texLRMaskUnpacked TEX(385)

ANF_KERNEL(anf_main_cs)
{
    int2 _62 = ivec2(((uint3)(GID_X, GID_Y, GID_Z)).xy);
    int _63 = _62.x;
    bool _69;
    if (_63 < AIFF_LR_OUTPUT_WIDTH)
    {
        _69 = _62.y < AIFF_LR_OUTPUT_HEIGHT;
    }
    else
    {
        _69 = false;
    }
    if (_69)
    {
        int2 _73 = g_clamp(_62, ivec2(0), ivec2(_50, _51));
        uint _80 = ((uint)((_73.y * AIFF_LR_OUTPUT_WIDTH) + _73.x) * 8u) >> 2u;
        uint _83 = _80 + 1u;
        imageStore(texLRFlowUnpacked, ivec2(((uint3)(GID_X, GID_Y, GID_Z)).xy), vec4(ivec4(g_clamp((int)(g_floor((unpackHalf2x16(bufFlowLR._m0[_80] & 65535u).x * 16384.0) + 0.5)), -32768, 32767), g_clamp((int)(g_floor((unpackHalf2x16((bufFlowLR._m0[_80] >> 16u) & 65535u).x * 16384.0) + 0.5)), -32768, 32767), g_clamp((int)(g_floor((unpackHalf2x16(bufFlowLR._m0[_83] & 65535u).x * 16384.0) + 0.5)), -32768, 32767), g_clamp((int)(g_floor((unpackHalf2x16((bufFlowLR._m0[_83] >> 16u) & 65535u).x * 16384.0) + 0.5)), -32768, 32767))) * 0.015625);
    }
    bool _131;
    if ((_63 & 1) == 0)
    {
        _131 = (_62.y & 1) == 0;
    }
    else
    {
        _131 = false;
    }
    if (_131)
    {
        int2 _134 = _62 / ivec2(2);
        bool _141;
        if (_134.x < AIFF_LR_MASK_OUTPUT_WIDTH)
        {
            _141 = _134.y < AIFF_LR_MASK_OUTPUT_HEIGHT;
        }
        else
        {
            _141 = false;
        }
        if (_141)
        {
            int2 _145 = g_clamp(_134, ivec2(0), ivec2(_54, _55));
            uint _150 = (uint)((_145.y * AIFF_LR_MASK_OUTPUT_WIDTH) + _145.x);
            imageStore(texLRMaskUnpacked, ivec2(uvec2(_134)), vec4((float)((uint)(g_clamp((int)(g_floor((unpackHalf2x16((bufMaskLR_1._m0[((_150 / 2u) * 4u) >> 2u] >> (((_150 & 1u) * 16u) & 31u)) & 65535u).x * 255.0) + 0.5)), 0, 255))) * 0.0039215688593685626983642578125));
        }
    }
}
