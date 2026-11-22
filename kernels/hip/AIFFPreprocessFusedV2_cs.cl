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
#define AIFF_QR_WIDTH ((int)SPECD(12, 0))
#define AIFF_QR_HEIGHT ((int)SPECD(13, 0))
#define _84 ((int)((AIFF_FULL_WIDTH - 1)))
#define _85 ((int)((AIFF_FULL_HEIGHT - 1)))
#define _90 ((int)((AIFF_PREPROCESS_WIDTH + 1)))
#define _91 ((int)((AIFF_PREPROCESS_HEIGHT + 1)))
#define _92 ((int)((AIFF_PREPROCESS_WIDTH - 1)))
#define _93 ((int)((AIFF_PREPROCESS_HEIGHT - 1)))
#define _94 ((int)((AIFF_FULL_WIDTH - 1)))
#define _95 ((int)((AIFF_FULL_HEIGHT - 1)))
typedef struct {
    uint _m0[];
} type_RWByteAddressBuffer_t;
#define projected_packed_mv_x (*(global type_RWByteAddressBuffer_t*)SLOT(386))
typedef struct {
    uint _m0[];
} projected_packed_mv_y_t;
#define projected_packed_mv_y_1 (*(global projected_packed_mv_y_t*)SLOT(387))
typedef struct {
    uint _m0[];
} depthStats_t;
#define depthStats_1 (*(global depthStats_t*)SLOT(388))
typedef struct {
    float2 jitter_prev_offsets;
    float2 jitter_curr_offsets;
    uint use_norm_depth;
    uint invert_depth;
} type_ConstantBuffer_PreprocessFusedV2PushConstants_t;
#define params (*(constant const type_ConstantBuffer_PreprocessFusedV2PushConstants_t*)a->push)
#define texRawMotion TEX(0)
#define texRawDepth TEX(1)
#define outputDepthPyramidQR TEX(384)
#define outputDilatedMV TEX(385)

ANF_KERNEL(anf_main_cs)
{
    local uint gs_tile_min;
    local uint gs_tile_max;
    local float gs_depth[34][34];
    do
    {
        int2 _106 = ivec2(((uint3)(GID_X, GID_Y, GID_Z)).xy);
        int _109 = (int)(((uint3)(LID_X, LID_Y, LID_Z)).y);
        int _113 = (int)(((uint3)(LID_X, LID_Y, LID_Z)).x);
        int _114 = (_109 * 16) + _113;
        bool _115 = _114 == 0;
        if (_115)
        {
            gs_tile_min = 2139095040u;
            gs_tile_max = 0u;
        }
        LDS_BARRIER();
        int2 _127 = ivec2((int)(((uint3)(WG_X, WG_Y, WG_Z)).x * 32u), (int)(((uint3)(WG_X, WG_Y, WG_Z)).y * 32u)) - ivec2(1);
        for (int _129 = _114; _129 < 1156; _129 += 256)
        {
            int _135 = _129 - 34 * (_129 / 34);
            int _136 = _129 / 34;
            int2 _138 = _127 + ivec2(_135, _136);
            int _139 = _138.x;
            bool _144;
            if (_139 >= 0)
            {
                _144 = _139 < AIFF_FULL_WIDTH;
            }
            else
            {
                _144 = false;
            }
            bool _149;
            if (_144)
            {
                _149 = _138.y >= 0;
            }
            else
            {
                _149 = false;
            }
            bool _154;
            if (_149)
            {
                _154 = _138.y < AIFF_FULL_HEIGHT;
            }
            else
            {
                _154 = false;
            }
            float4 _159 = texelFetch(texRawDepth, ivec2(uvec2(g_clamp(_138, ivec2(0), ivec2(_84, _85)))), (int)(0u));
            float _160 = _159.x;
            float _171;
            if (params.invert_depth != 0u)
            {
                _171 = 1.0 / ((1000.0 * _160) + (0.300000011920928955078125 * (1.0 - _160)));
            }
            else
            {
                _171 = _160;
            }
            if (_154)
            {
                uint _174 = floatBitsToUint(_171);
                uint _175 = atomicMin(&(gs_tile_min), _174);
                uint _176 = atomicMax(&(gs_tile_max), _174);
            }
            gs_depth[_136][_135] = _171;
        }
        LDS_BARRIER();
        if (_115)
        {
            uint _182 = atomicMin(&(depthStats_1._m0[2u]), gs_tile_min);
            uint _185 = atomicMax(&(depthStats_1._m0[3u]), gs_tile_max);
        }
        int _186 = _106.x;
        bool _192;
        if (_186 < AIFF_QR_WIDTH)
        {
            _192 = _106.y >= AIFF_QR_HEIGHT;
        }
        else
        {
            _192 = true;
        }
        if (_192)
        {
            break;
        }
        bool _200;
        if (_186 < AIFF_PREPROCESS_WIDTH)
        {
            _200 = _106.y < AIFF_PREPROCESS_HEIGHT;
        }
        else
        {
            _200 = false;
        }
        int _201 = _90 / 2;
        bool _210;
        if (_186 < (_201 * 2))
        {
            int _207 = _91 / 2;
            _210 = _106.y >= (_207 * 2);
        }
        else
        {
            _210 = true;
        }
        if (_210)
        {
            imageStore(outputDepthPyramidQR, ivec2(((uint3)(GID_X, GID_Y, GID_Z)).xy), vec4(0.0));
            break;
        }
        int2 _215 = g_min(_106, ivec2(_92, _93));
        int _221 = ((_113 * 2) + 1) + ((_215.x - _186) * 2);
        int _225 = _106.y;
        int _228 = ((_109 * 2) + 1) + ((_215.y - _225) * 2);
        int2 _229 = ivec2(_221, _228);
        int _230 = _228 - 1;
        int _231 = _221 - 1;
        bool _234 = gs_depth[_230][_231] > (-100000002004087734272.0);
        int2 _238;
        if (_234)
        {
            _238 = ivec2(_231, _230);
        }
        else
        {
            _238 = _229;
        }
        float _239 = _234 ? gs_depth[_230][_231] : (-100000002004087734272.0);
        bool _242 = gs_depth[_230][_221] > _239;
        int2 _246;
        if (_242)
        {
            _246 = ivec2(_221, _230);
        }
        else
        {
            _246 = _238;
        }
        float _247 = _242 ? gs_depth[_230][_221] : _239;
        int _248 = _221 + 1;
        bool _251 = gs_depth[_230][_248] > _247;
        int2 _255;
        if (_251)
        {
            _255 = ivec2(_248, _230);
        }
        else
        {
            _255 = _246;
        }
        float _256 = _251 ? gs_depth[_230][_248] : _247;
        int _257 = _221 + 2;
        bool _260 = gs_depth[_230][_257] > _256;
        int2 _264;
        if (_260)
        {
            _264 = ivec2(_257, _230);
        }
        else
        {
            _264 = _255;
        }
        float _265 = _260 ? gs_depth[_230][_257] : _256;
        bool _268 = gs_depth[_228][_231] > _265;
        int2 _272;
        if (_268)
        {
            _272 = ivec2(_231, _228);
        }
        else
        {
            _272 = _264;
        }
        float _273 = _268 ? gs_depth[_228][_231] : _265;
        bool _276 = gs_depth[_228][_221] > _273;
        float _277 = _276 ? gs_depth[_228][_221] : _273;
        bool _282 = gs_depth[_228][_248] > _277;
        int2 _286;
        if (_282)
        {
            _286 = ivec2(_248, _228);
        }
        else
        {
            _286 = g_mix(_272, _229, bvec2(_276));
        }
        float _287 = _282 ? gs_depth[_228][_248] : _277;
        bool _290 = gs_depth[_228][_257] > _287;
        int2 _294;
        if (_290)
        {
            _294 = ivec2(_257, _228);
        }
        else
        {
            _294 = _286;
        }
        float _295 = _290 ? gs_depth[_228][_257] : _287;
        int _296 = _228 + 1;
        bool _299 = gs_depth[_296][_231] > _295;
        int2 _303;
        if (_299)
        {
            _303 = ivec2(_231, _296);
        }
        else
        {
            _303 = _294;
        }
        float _304 = _299 ? gs_depth[_296][_231] : _295;
        bool _307 = gs_depth[_296][_221] > _304;
        int2 _311;
        if (_307)
        {
            _311 = ivec2(_221, _296);
        }
        else
        {
            _311 = _303;
        }
        float _312 = _307 ? gs_depth[_296][_221] : _304;
        bool _315 = gs_depth[_296][_248] > _312;
        int2 _319;
        if (_315)
        {
            _319 = ivec2(_248, _296);
        }
        else
        {
            _319 = _311;
        }
        float _320 = _315 ? gs_depth[_296][_248] : _312;
        bool _323 = gs_depth[_296][_257] > _320;
        int2 _327;
        if (_323)
        {
            _327 = ivec2(_257, _296);
        }
        else
        {
            _327 = _319;
        }
        float _328 = _323 ? gs_depth[_296][_257] : _320;
        int _329 = _228 + 2;
        bool _332 = gs_depth[_329][_231] > _328;
        int2 _336;
        if (_332)
        {
            _336 = ivec2(_231, _329);
        }
        else
        {
            _336 = _327;
        }
        float _337 = _332 ? gs_depth[_329][_231] : _328;
        bool _340 = gs_depth[_329][_221] > _337;
        int2 _344;
        if (_340)
        {
            _344 = ivec2(_221, _329);
        }
        else
        {
            _344 = _336;
        }
        float _345 = _340 ? gs_depth[_329][_221] : _337;
        bool _348 = gs_depth[_329][_248] > _345;
        int2 _352;
        if (_348)
        {
            _352 = ivec2(_248, _329);
        }
        else
        {
            _352 = _344;
        }
        float _353 = _348 ? gs_depth[_329][_248] : _345;
        bool _356 = gs_depth[_329][_257] > _353;
        int2 _360;
        if (_356)
        {
            _360 = ivec2(_257, _329);
        }
        else
        {
            _360 = _352;
        }
        float _361 = _356 ? gs_depth[_329][_257] : _353;
        float2 _374 = texelFetch(texRawMotion, ivec2(uvec2(g_clamp(_127 + _360, ivec2(0), ivec2(_94, _95)))), (int)(0u)).xy + (params.jitter_prev_offsets - params.jitter_curr_offsets);
        float _375 = (float)(AIFF_PREPROCESS_WIDTH);
        float _379 = (float)(AIFF_PREPROCESS_HEIGHT);
        float _403;
        if (params.use_norm_depth != 0u)
        {
            float _391 = uintBitsToFloat(depthStats_1._m0[0u]);
            float _395 = uintBitsToFloat(depthStats_1._m0[1u]) - _391;
            float _402;
            if (_395 > 9.9999999747524270787835121154785e-07)
            {
                _402 = (_361 - _391) / (_395 + 9.9999999747524270787835121154785e-07);
            }
            else
            {
                _402 = _361;
            }
            _403 = _402;
        }
        else
        {
            _403 = _361;
        }
        float _404 = g_clamp(_403, 0.0, 1.0);
        imageStore(outputDepthPyramidQR, ivec2(((uint3)(GID_X, GID_Y, GID_Z)).xy), vec4(_404));
        if (!_200)
        {
            break;
        }
        float2 _409 = vec2(_374.x * (-_375), _374.y * (-_379)) * 0.5;
        imageStore(outputDilatedMV, ivec2(((uint3)(GID_X, GID_Y, GID_Z)).xy), _409.xyyy);
        int _415 = _186 + (int)(g_floor(_409.x + 0.5));
        int _420 = _225 + (int)(g_floor(_409.y + 0.5));
        bool _425;
        if (_415 >= 0)
        {
            _425 = _415 < AIFF_PREPROCESS_WIDTH;
        }
        else
        {
            _425 = false;
        }
        bool _429;
        if (_425)
        {
            _429 = _420 >= 0;
        }
        else
        {
            _429 = false;
        }
        bool _433;
        if (_429)
        {
            _433 = _420 < AIFF_PREPROCESS_HEIGHT;
        }
        else
        {
            _433 = false;
        }
        if (_433)
        {
            float2 _436 = g_clamp(_409, vec2(-255.9921875), vec2(255.9921875));
            float _437 = _436.x;
            int _439 = (int)(_437 * 128.0);
            float _440 = _436.y;
            int _442 = (int)(_440 * 128.0);
            int _447;
            if (_437 < 0.0)
            {
                _447 = 32768 - _439;
            }
            else
            {
                _447 = _439;
            }
            int _452;
            if (_440 < 0.0)
            {
                _452 = 32768 - _442;
            }
            else
            {
                _452 = _442;
            }
            uint _456 = 2147483648u | ((uint)(_404 * 16384.0) << 16u);
            uint _467 = ((((uint)(_420) * (uint)(AIFF_PREPROCESS_WIDTH)) + (uint)(_415)) * 4u) >> 2u;
            uint _469 = atomicMax(&(projected_packed_mv_x._m0[_467]), _456 | (uint)(_447));
            uint _471 = atomicMax(&(projected_packed_mv_y_1._m0[_467]), _456 | (uint)(_452));
        }
        break;
    } while(false);
}
