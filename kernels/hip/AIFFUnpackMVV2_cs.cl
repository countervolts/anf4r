// anf4r: OpenCL C port of Qualcomm's main_cs pass, translated from SPIRV-Cross GLSL by tools/glsl2cl.py.
// SPDX-License-Identifier: GPL-3.0-only
#define LX 16
#define LY 16
#define LZ 1
#include "anf_port.h"

#define AIFF_PREPROCESS_WIDTH ((int)SPECD(2, 0))
#define AIFF_PREPROCESS_HEIGHT ((int)SPECD(3, 0))
typedef struct {
    uint _m0[];
} type_ByteAddressBuffer_t;
#define packed_data_x (*(global type_ByteAddressBuffer_t*)SLOT(1))
typedef struct {
    uint _m0[];
} packed_data_y_t;
#define packed_data_y_1 (*(global packed_data_y_t*)SLOT(2))
#define texCollocatedMVProj TEX(0)
#define fwd_mvs TEX(384)
#define bwd_mvs TEX(385)
#define depth_res TEX(386)
#define mask_res TEX(387)

ANF_KERNEL(anf_main_cs)
{
    do
    {
        int2 _56 = ivec2(((uint3)(GID_X, GID_Y, GID_Z)).xy);
        int _57 = _56.x;
        bool _63;
        if (_57 < AIFF_PREPROCESS_WIDTH)
        {
            _63 = _56.y >= AIFF_PREPROCESS_HEIGHT;
        }
        else
        {
            _63 = true;
        }
        if (_63)
        {
            break;
        }
        uint _73 = ((((uint)(_56.y) * (uint)(AIFF_PREPROCESS_WIDTH)) + (uint)(_57)) * 4u) >> 2u;
        uint _75 = packed_data_x._m0[_73];
        int _79 = (int)(_75 & 65535u);
        int _81 = (int)(packed_data_y_1._m0[_73] & 65535u);
        int _88;
        if (_79 > 32767)
        {
            _88 = -(_79 & 32767);
        }
        else
        {
            _88 = _79;
        }
        int _95;
        if (_81 > 32767)
        {
            _95 = -(_81 & 32767);
        }
        else
        {
            _95 = _81;
        }
        uint _103 = (_75 >> 31u) & 1u;
        int2 _120;
        if (_103 != 0u)
        {
            _120 = ivec2(_88, _95);
        }
        else
        {
            float4 _110 = texelFetch(texCollocatedMVProj, ivec2(((uint3)(GID_X, GID_Y, GID_Z)).xy), (int)(0u));
            _120 = ivec2((int)(g_clamp(_110.x, -255.9921875, 255.9921875) * 128.0), (int)(g_clamp(_110.y, -255.9921875, 255.9921875) * 128.0));
        }
        imageStoreI(fwd_mvs, ivec2(((uint3)(GID_X, GID_Y, GID_Z)).xy), _120.xyyy);
        imageStoreI(bwd_mvs, ivec2(((uint3)(GID_X, GID_Y, GID_Z)).xy), (-_120).xyyy);
        imageStoreU(depth_res, ivec2(((uint3)(GID_X, GID_Y, GID_Z)).xy), uvec4((uint)(g_clamp((float)((_75 >> 16u) & 32767u) * 0.01556396484375, 0.0, 255.0)) * _103));
        imageStoreU(mask_res, ivec2(((uint3)(GID_X, GID_Y, GID_Z)).xy), uvec4(_103 * 255u));
        break;
    } while(false);
}
