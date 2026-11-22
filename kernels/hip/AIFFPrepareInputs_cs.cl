// anf4r: OpenCL C port of Qualcomm's main_cs pass, translated from SPIRV-Cross GLSL by tools/glsl2cl.py.
// SPDX-License-Identifier: GPL-3.0-only
#define LX 16
#define LY 16
#define LZ 1
#include "anf_port.h"

typedef struct {
    float2 jitter_prev_offsets;
    float2 jitter_curr_offsets;
    int block_size;
    float scaling_factor;
    uint yMax;
} type_ConstantBuffer_DepthDilatePushConstants_t;
#define params (*(constant const type_ConstantBuffer_DepthDilatePushConstants_t*)a->push)
#define texRawMotion TEX(0)
#define texRawDepth TEX(1)
#define outputPreparedMotion TEX(384)
#define outputPreparedDepth TEX(385)

ANF_KERNEL(anf_main_cs)
{
    do
    {
        int2 _39 = ivec2(((uint3)(GID_X, GID_Y, GID_Z)).xy);
        uint2 _41 = uvec2(textureSize(texRawMotion, 0));
        int _43 = (int)(_41.x);
        int _45 = (int)(_41.y);
        int _46 = _39.x;
        bool _52;
        if (_46 < _43)
        {
            _52 = _39.y >= _45;
        }
        else
        {
            _52 = true;
        }
        if (_52)
        {
            break;
        }
        int2 _57 = ivec3(_46, _39.y, 0).xy;
        float2 _66 = texelFetch(texRawMotion, _57, 0).xy + (params.jitter_prev_offsets - params.jitter_curr_offsets);
        imageStore(outputPreparedMotion, ivec2(((uint3)(GID_X, GID_Y, GID_Z)).xy), vec2(_66.x * (-(float)(_43)), _66.y * (-(float)(_45))).xyyy);
        imageStore(outputPreparedDepth, ivec2(((uint3)(GID_X, GID_Y, GID_Z)).xy), vec4(texelFetch(texRawDepth, _57, 0).x));
        break;
    } while(false);
}
