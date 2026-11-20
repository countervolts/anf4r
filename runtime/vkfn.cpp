// SPDX-License-Identifier: GPL-3.0-only
#include "vkfn.h"

#ifdef ANF4R_DYNAMIC_VK
#define ANF4R_VK_DEFINE(f) PFN_##f f = nullptr;
PFN_vkCreateInstance vkCreateInstance = nullptr;
ANF4R_VK_INSTANCE_FUNCS(ANF4R_VK_DEFINE)
ANF4R_VK_DEVICE_FUNCS(ANF4R_VK_DEFINE)
#undef ANF4R_VK_DEFINE
#endif

namespace anf4r {
PFN_vkGetInstanceProcAddr g_gipa = nullptr;
bool loadVulkan(PFN_vkGetInstanceProcAddr gipa, VkInstance instance, VkDevice device) {
    g_gipa = gipa;
#ifdef ANF4R_DYNAMIC_VK
    bool ok = true;
#define LI(f) f = reinterpret_cast<PFN_##f>(gipa(instance, #f)); ok &= f != nullptr;
    ANF4R_VK_INSTANCE_FUNCS(LI)
#undef LI
#define LD(f) f = reinterpret_cast<PFN_##f>(vkGetDeviceProcAddr(device, #f)); ok &= f != nullptr;
    ANF4R_VK_DEVICE_FUNCS(LD)
#undef LD
    return ok;
#else
    (void)gipa; (void)instance; (void)device;
    return true;
#endif
}
}
