// The upscaler behind the OptiScaler plugin (asi.cpp, framegen.cpp): anf4r, Qualcomm's Adreno Neural Fusion. The plugin's other
// files are m4r's (github.com/countervolts/m4r, plugin/, GPL-3.0-only, same author), kept identical; m4r's copy of this header
// names m4r's API, which has the same shape as anf4r's.
// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include "anf4r.h"

#define BE_NAME "anf4r"                // <plugin dir>\anf4r\ holds the data, the log is anf4r.log there
#define BE_DATA "data"                 // modelzoo\ (the ANF weights) and hip\ (the passes' code objects for this GPU)
#define BE_UPSCALER "ANF"              // provider names in OptiScaler's lists
#define BE_FG "ANF FG"
#define BE_VERSION_ID 0x616e66340001ull   // "anf4" 1
#define BE_VERSION_NAME "1.0.0 (Adreno Neural Fusion, anf4r)"

typedef anf4r_context be_context;
typedef anf4r_upscale_desc be_upscale_desc;
typedef anf4r_interpolate_desc be_interpolate_desc;
#define be_create anf4r_create
#define be_destroy anf4r_destroy
#define be_last_error anf4r_last_error
#define be_upscale_init anf4r_upscale_init
#define be_upscale anf4r_upscale
#define be_interpolate_init anf4r_interpolate_init
#define be_interpolate anf4r_interpolate
