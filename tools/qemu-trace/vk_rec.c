// anf4r recording libvulkan.so: lets libanf initialize + create/dispatch against fake handles, logging the calls
// that reveal the pipeline (shader modules, dispatches, barriers, push constants, descriptor writes, resources).
// Everything returns VK_SUCCESS with non-null opaque handles; nothing executes on a GPU.
#include <vulkan/vulkan.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include "log.h"
static uint64_t g_h = 0x7000;
static void* H(){ return (void*)(uintptr_t)(++g_h); }

// ---- device capability reporting (so IsTechniqueSupported passes) ----
VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceProperties(VkPhysicalDevice pd, VkPhysicalDeviceProperties* p){
    memset(p,0,sizeof *p); p->apiVersion=VK_API_VERSION_1_3; p->vendorID=0x5143/*Qualcomm*/; p->deviceID=0x840;
    p->deviceType=VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU; strcpy(p->deviceName,"Adreno (TM) 840");
    p->limits.maxComputeWorkGroupInvocations=1024; p->limits.maxComputeSharedMemorySize=32768;
    for(int i=0;i<3;i++){p->limits.maxComputeWorkGroupCount[i]=65535;p->limits.maxComputeWorkGroupSize[i]=1024;}
    p->limits.maxPushConstantsSize=256; p->limits.maxBoundDescriptorSets=8; p->limits.maxImageDimension2D=16384;
    p->limits.minStorageBufferOffsetAlignment=16; p->limits.minUniformBufferOffsetAlignment=16;
    p->limits.maxPerStageDescriptorStorageBuffers=256; p->limits.maxPerStageResources=512;
    LOG("vkGetPhysicalDeviceProperties -> Adreno 840"); }
VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceProperties2(VkPhysicalDevice pd, VkPhysicalDeviceProperties2* p){
    vkGetPhysicalDeviceProperties(pd,&p->properties);
    for(VkBaseOutStructure* s=(VkBaseOutStructure*)p->pNext; s; s=s->pNext){ /* leave subprops zeroed */ } }
VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFeatures(VkPhysicalDevice pd, VkPhysicalDeviceFeatures* f){ memset(f,0xFF,sizeof *f); }
static void fill_bools(void* st, int n){ VkBool32* b=(VkBool32*)((char*)st+16); for(int i=0;i<n;i++) b[i]=VK_TRUE; }
VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFeatures2(VkPhysicalDevice pd, VkPhysicalDeviceFeatures2* f){
    memset(&f->features,0xFF,sizeof f->features);
    for(VkBaseOutStructure* s=(VkBaseOutStructure*)f->pNext; s; s=s->pNext){
        int n=2; // default: only the first 1-2 bools (safe for minimal structs)
        switch((int)s->sType){
          case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES: n=12; break;
          case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES: n=47; break;
          case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES: n=15; break;
          default: n=2; break;
        }
        fill_bools(s,n);
        LOG("vkGetPhysicalDeviceFeatures2 pNext sType=%d (+%d bools)", (int)s->sType, n);
    }
    LOG("vkGetPhysicalDeviceFeatures2"); }
VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceMemoryProperties(VkPhysicalDevice pd, VkPhysicalDeviceMemoryProperties* m){
    memset(m,0,sizeof *m); m->memoryTypeCount=1; m->memoryHeapCount=1; m->memoryHeaps[0].size=1ull<<32;
    m->memoryHeaps[0].flags=VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
    m->memoryTypes[0].heapIndex=0; m->memoryTypes[0].propertyFlags=VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT|VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT; }
VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceMemoryProperties2(VkPhysicalDevice pd, VkPhysicalDeviceMemoryProperties2* m){ vkGetPhysicalDeviceMemoryProperties(pd,&m->memoryProperties); }
VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceQueueFamilyProperties(VkPhysicalDevice pd, uint32_t* n, VkQueueFamilyProperties* q){
    if(!q){*n=1;return;} if(*n){ q[0].queueFlags=VK_QUEUE_GRAPHICS_BIT|VK_QUEUE_COMPUTE_BIT|VK_QUEUE_TRANSFER_BIT; q[0].queueCount=4; *n=1; } }
VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceQueueFamilyProperties2(VkPhysicalDevice pd, uint32_t* n, VkQueueFamilyProperties2* q){
    if(!q){*n=1;return;} if(*n){ q[0].queueFamilyProperties.queueFlags=VK_QUEUE_GRAPHICS_BIT|VK_QUEUE_COMPUTE_BIT|VK_QUEUE_TRANSFER_BIT; q[0].queueFamilyProperties.queueCount=4; *n=1; } }
VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFormatProperties(VkPhysicalDevice pd, VkFormat f, VkFormatProperties* p){
    p->linearTilingFeatures=p->optimalTilingFeatures=p->bufferFeatures=0x7FFFFFFF; }
VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFormatProperties2(VkPhysicalDevice pd, VkFormat f, VkFormatProperties2* p){ vkGetPhysicalDeviceFormatProperties(pd,f,&p->formatProperties); }
VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceImageFormatProperties(VkPhysicalDevice pd,VkFormat f,VkImageType t,VkImageTiling ti,VkImageUsageFlags u,VkImageCreateFlags c,VkImageFormatProperties* p){ memset(p,0,sizeof*p); p->maxExtent.width=p->maxExtent.height=16384; p->maxExtent.depth=1; p->maxMipLevels=1; p->maxArrayLayers=1; p->sampleCounts=VK_SAMPLE_COUNT_1_BIT; p->maxResourceSize=1ull<<31; return VK_SUCCESS; }
VKAPI_ATTR VkResult VKAPI_CALL vkEnumeratePhysicalDevices(VkInstance i, uint32_t* n, VkPhysicalDevice* d){ if(!d){*n=1;return VK_SUCCESS;} if(*n){d[0]=(VkPhysicalDevice)(uintptr_t)0x1002;*n=1;} return VK_SUCCESS; }
VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateDeviceExtensionProperties(VkPhysicalDevice pd,const char* l,uint32_t* n,VkExtensionProperties* p){
    static const char* exts[]={"VK_KHR_cooperative_matrix","VK_QCOM_tile_memory_heap","VK_ARM_tensors","VK_ARM_data_graph","VK_KHR_shader_float_controls","VK_EXT_subgroup_size_control","VK_KHR_8bit_storage","VK_KHR_16bit_storage","VK_KHR_external_memory_fd","VK_KHR_shader_integer_dot_product","VK_QCOM_cooperative_matrix_conversion","VK_EXT_external_memory_dma_buf"};
    uint32_t cnt=sizeof exts/sizeof*exts; if(!p){*n=cnt;return VK_SUCCESS;} uint32_t m=*n<cnt?*n:cnt; for(uint32_t i=0;i<m;i++){memset(&p[i],0,sizeof p[i]);strncpy(p[i].extensionName,exts[i],sizeof p[i].extensionName-1);p[i].specVersion=1;} *n=m; return VK_SUCCESS; }
VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceExtensionProperties(const char* l,uint32_t* n,VkExtensionProperties* p){ if(!p){*n=0;return VK_SUCCESS;} *n=0; return VK_SUCCESS; }

// ---- object creation: return non-null handles ----
#define MK(ret,name,...) VKAPI_ATTR VkResult VKAPI_CALL name(__VA_ARGS__)
VKAPI_ATTR VkResult VKAPI_CALL vkCreateInstance(const VkInstanceCreateInfo* ci,const VkAllocationCallbacks* a,VkInstance* o){ *o=(VkInstance)H(); LOG("vkCreateInstance"); return VK_SUCCESS; }
VKAPI_ATTR VkResult VKAPI_CALL vkCreateDevice(VkPhysicalDevice pd,const VkDeviceCreateInfo* ci,const VkAllocationCallbacks* a,VkDevice* o){ *o=(VkDevice)H();
    for(uint32_t i=0;i<ci->enabledExtensionCount;i++) LOG("vkCreateDevice ext: %s", ci->ppEnabledExtensionNames[i]); return VK_SUCCESS; }
VKAPI_ATTR void VKAPI_CALL vkGetDeviceQueue(VkDevice d,uint32_t f,uint32_t i,VkQueue* q){ *q=(VkQueue)H(); }
VKAPI_ATTR VkResult VKAPI_CALL vkCreateShaderModule(VkDevice d,const VkShaderModuleCreateInfo* ci,const VkAllocationCallbacks* a,VkShaderModule* o){
    *o=(VkShaderModule)H(); LOG("vkCreateShaderModule size=%zu", ci->codeSize);
    const char* dd=getenv("ANF_TRACE_DUMP"); if(dd){ char p[512]; snprintf(p,sizeof p,"%s/mod_%llx.spv",dd,(unsigned long long)(uintptr_t)*o); FILE* f=fopen(p,"wb"); if(f){fwrite(ci->pCode,1,ci->codeSize,f);fclose(f);} } return VK_SUCCESS; }
VKAPI_ATTR VkResult VKAPI_CALL vkCreateComputePipelines(VkDevice d,VkPipelineCache c,uint32_t n,const VkComputePipelineCreateInfo* ci,const VkAllocationCallbacks* a,VkPipeline* o){ for(uint32_t i=0;i<n;i++){o[i]=(VkPipeline)H(); LOG("vkCreateComputePipelines stageName=%s", ci[i].stage.pName?ci[i].stage.pName:"?");} return VK_SUCCESS; }
VKAPI_ATTR void VKAPI_CALL vkCmdBindPipeline(VkCommandBuffer cb,VkPipelineBindPoint bp,VkPipeline p){ LOG("vkCmdBindPipeline %p", (void*)p); }
VKAPI_ATTR void VKAPI_CALL vkCmdDispatch(VkCommandBuffer cb,uint32_t x,uint32_t y,uint32_t z){ LOG("vkCmdDispatch %u %u %u", x,y,z); }
VKAPI_ATTR void VKAPI_CALL vkCmdPushConstants(VkCommandBuffer cb,VkPipelineLayout l,VkShaderStageFlags s,uint32_t off,uint32_t sz,const void* v){
    char hex[256]=""; const uint8_t* b=v; for(uint32_t i=0;i<sz&&i<48;i++){char t[4];snprintf(t,4,"%02x",b[i]);strcat(hex,t);} LOG("vkCmdPushConstants off=%u sz=%u %s",off,sz,hex); }
VKAPI_ATTR void VKAPI_CALL vkCmdPipelineBarrier(VkCommandBuffer cb,VkPipelineStageFlags s,VkPipelineStageFlags d,VkDependencyFlags df,uint32_t mb,const VkMemoryBarrier* pmb,uint32_t bb,const VkBufferMemoryBarrier* pbb,uint32_t ib,const VkImageMemoryBarrier* pib){ LOG("vkCmdPipelineBarrier mem=%u buf=%u img=%u",mb,bb,ib); }
VKAPI_ATTR void VKAPI_CALL vkUpdateDescriptorSets(VkDevice d,uint32_t n,const VkWriteDescriptorSet* w,uint32_t c,const VkCopyDescriptorSet* cp){ for(uint32_t i=0;i<n;i++) LOG("  descWrite binding=%u type=%u count=%u", w[i].dstBinding, w[i].descriptorType, w[i].descriptorCount); }
VKAPI_ATTR VkResult VKAPI_CALL vkCreateBuffer(VkDevice d,const VkBufferCreateInfo* ci,const VkAllocationCallbacks* a,VkBuffer* o){ *o=(VkBuffer)H(); LOG("vkCreateBuffer size=%llu usage=0x%x",(unsigned long long)ci->size,ci->usage); return VK_SUCCESS; }
VKAPI_ATTR VkResult VKAPI_CALL vkCreateImage(VkDevice d,const VkImageCreateInfo* ci,const VkAllocationCallbacks* a,VkImage* o){ *o=(VkImage)H(); LOG("vkCreateImage %ux%u fmt=%d usage=0x%x",ci->extent.width,ci->extent.height,ci->format,ci->usage); return VK_SUCCESS; }
VKAPI_ATTR void VKAPI_CALL vkGetBufferMemoryRequirements(VkDevice d,VkBuffer b,VkMemoryRequirements* m){ m->size=1<<20;m->alignment=256;m->memoryTypeBits=1; }
VKAPI_ATTR void VKAPI_CALL vkGetImageMemoryRequirements(VkDevice d,VkImage im,VkMemoryRequirements* m){ m->size=1<<22;m->alignment=256;m->memoryTypeBits=1; }
VKAPI_ATTR VkResult VKAPI_CALL vkAllocateMemory(VkDevice d,const VkMemoryAllocateInfo* ai,const VkAllocationCallbacks* a,VkDeviceMemory* o){ *o=(VkDeviceMemory)H(); return VK_SUCCESS; }
VKAPI_ATTR VkResult VKAPI_CALL vkMapMemory(VkDevice d,VkDeviceMemory m,VkDeviceSize off,VkDeviceSize sz,VkMemoryMapFlags f,void** pp){ *pp=calloc(1, sz==VK_WHOLE_SIZE?(1<<20):sz); return VK_SUCCESS; }

// ---- generic success stub for everything else ----
VKAPI_ATTR VkResult VKAPI_CALL generic_ok(){ return VK_SUCCESS; }

typedef PFN_vkVoidFunction (*gipa_t)(VkInstance, const char*);
PFN_vkVoidFunction vkGetInstanceProcAddr(VkInstance i, const char* name);
PFN_vkVoidFunction vkGetDeviceProcAddr(VkDevice d, const char* name){ return vkGetInstanceProcAddr((VkInstance)0, name); }

struct E { const char* n; PFN_vkVoidFunction f; };
#define E(fn) { #fn, (PFN_vkVoidFunction)fn }
PFN_vkVoidFunction vkGetInstanceProcAddr(VkInstance inst, const char* name){
    static const struct E tbl[] = {
        E(vkGetInstanceProcAddr),E(vkGetDeviceProcAddr),E(vkCreateInstance),E(vkCreateDevice),E(vkEnumeratePhysicalDevices),
        E(vkGetPhysicalDeviceProperties),E(vkGetPhysicalDeviceProperties2),E(vkGetPhysicalDeviceFeatures),E(vkGetPhysicalDeviceFeatures2),
        E(vkGetPhysicalDeviceMemoryProperties),E(vkGetPhysicalDeviceMemoryProperties2),E(vkGetPhysicalDeviceQueueFamilyProperties),
        E(vkGetPhysicalDeviceQueueFamilyProperties2),E(vkGetPhysicalDeviceFormatProperties),E(vkGetPhysicalDeviceFormatProperties2),
        E(vkGetPhysicalDeviceImageFormatProperties),E(vkEnumerateDeviceExtensionProperties),E(vkEnumerateInstanceExtensionProperties),
        E(vkGetDeviceQueue),E(vkCreateShaderModule),E(vkCreateComputePipelines),E(vkCmdBindPipeline),E(vkCmdDispatch),
        E(vkCmdPushConstants),E(vkCmdPipelineBarrier),E(vkUpdateDescriptorSets),E(vkCreateBuffer),E(vkCreateImage),
        E(vkGetBufferMemoryRequirements),E(vkGetImageMemoryRequirements),E(vkAllocateMemory),E(vkMapMemory),
    };
    for(unsigned k=0;k<sizeof tbl/sizeof*tbl;k++) if(!strcmp(tbl[k].n,name)) return tbl[k].f;
    // handle-returning creators we didn't special-case: return a stub that writes a handle into the last out-param.
    LOG("vkGetInstanceProcAddr(generic) %s", name);
    return (PFN_vkVoidFunction)generic_ok;
}
