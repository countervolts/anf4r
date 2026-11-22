// anf4r's Vulkan function table (vkfn.h), as hipvk::install() fills it. hipvk is m4r's (github.com/countervolts/m4r,
// runtime/hipvk, GPL-3.0-only, same author): the other files of this folder are kept identical to m4r's.
// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include "../vkfn.h"

#define HIPVK_TABLE(X) X(vkCreateInstance) ANF4R_VK_INSTANCE_FUNCS(X) ANF4R_VK_DEVICE_FUNCS(X)
