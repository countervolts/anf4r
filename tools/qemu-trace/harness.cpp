// anf4r trace harness (aarch64/bionic): drive libanf.so's SR create+dispatch so the shims capture the graph.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <stdint.h>
#include "../../vendor/anf-sdk/public/anf.h"
#include "../../vendor/anf-sdk/public/anf_types_vk.h"
#include "../../vendor/anf-sdk/public/anf_sr.h"

static void logcb(AnfLogLevel lvl, const char* m){ fprintf(stderr, "[anf %d] %s\n", lvl, m); }
#define CK(expr) do{ AnfResult _r=(expr); fprintf(stderr,"==> %s = %d\n", #expr, _r);}while(0)

// fake Vulkan handles (the vk shim keys off non-null pointers)
static uint64_t fakeInstance=0x1001, fakePhys=0x1002, fakeDev=0x1003, fakeQueue=0x1004;

int main(int argc, char** argv){
    const char* lib = argc>1?argv[1]:"libanf.so";
    void* h = dlopen(lib, RTLD_NOW|RTLD_GLOBAL);
    if(!h){ fprintf(stderr,"dlopen %s: %s\n", lib, dlerror()); return 1; }
    void (*GetAnfFunctions)(AnfFunctions*) = (void(*)(AnfFunctions*))dlsym(h, "GetAnfFunctions");
    if(!GetAnfFunctions){ fprintf(stderr,"no GetAnfFunctions: %s\n", dlerror()); return 1; }
    AnfFunctions anf; memset(&anf,0,sizeof anf);
    GetAnfFunctions(&anf);
    fprintf(stderr,"GetAnfFunctions ok; CreateInstance=%p DispatchTechnique=%p\n",(void*)anf.CreateInstance,(void*)anf.DispatchTechnique);

    AnfVersion ver; if(anf.QueryAnfVersion){ anf.QueryAnfVersion(&ver); fprintf(stderr,"ANF version %u.%u.%u\n",ver.major,ver.minor,ver.build); }

    AnfInstanceCreateInfo ici; memset(&ici,0,sizeof ici);
    ici.header.type=ANF_STYPE_INSTANCE_CREATE_INFO; ici.clientApi=ANF_CLIENT_API_VULKAN;
    ici.logLevel=ANF_LOG_LEVEL_INFO; ici.logMessageCallback=logcb;
    AnfInstance inst=0; CK(anf.CreateInstance(&ici,&inst));
    if(!inst){ fprintf(stderr,"no instance\n"); return 2; }

    uint32_t nt=0; AnfTechniqueId* ids=0;

    AnfTechniqueId tid = ANF_TECHNIQUE_ID_SR_TEMPORAL;
    const AnfTechniqueRequirements* req=0;
    if(anf.QueryTechniqueRequirements){ CK(anf.QueryTechniqueRequirements(inst,tid,&req));
        if(req){ fprintf(stderr,"SR requires %u resources; clientApiReq type=%d\n", req->numRequiredResources, req->pClientApiRequirements?(int)req->pClientApiRequirements->type:-1);
            const AnfTechniqueVulkanApiRequirements* vr=(const AnfTechniqueVulkanApiRequirements*)req->pClientApiRequirements;
            if(vr){ for(uint32_t i=0;i<vr->numInstanceExtensions;++i)fprintf(stderr,"  inst ext: %s\n",vr->ppInstanceExtensions[i]);
                    for(uint32_t i=0;i<vr->numDeviceExtensions;++i)fprintf(stderr,"  dev ext:  %s\n",vr->ppDeviceExtensions[i]);
                    fprintf(stderr,"  queueFlags=0x%x\n",vr->techniqueQueueFlags);} }
    }
    AnfAdapterInfoVulkan ad; memset(&ad,0,sizeof ad);
    ad.header.type=ANF_STYPE_ADAPTER_INFO_VULKAN; ad.instance=(VkInstance)fakeInstance; ad.physicalDevice=(VkPhysicalDevice)fakePhys; ad.device=(VkDevice)fakeDev;
    if(anf.IsTechniqueSupported) CK(anf.IsTechniqueSupported(inst,tid,&ad.header));

    // resource requirements
    if(anf.QueryTechniqueResourceRequirements){ for(uint32_t l=0;l<4;++l){ const AnfResourceRequirements* rr=0;
        AnfResult r=anf.QueryTechniqueResourceRequirements(inst,tid,(AnfResourceLabel)l,&rr);
        if(r==ANF_RESULT_SUCCESS&&rr){ fprintf(stderr,"label %u: type=%d formats=%u\n",l,rr->resourceType,rr->numSupportedFormats);} } }

    // register device + queue
    AnfQueueInfoVulkan qi; memset(&qi,0,sizeof qi); qi.header.type=ANF_STYPE_QUEUE_INFO_VULKAN;
    VkQueue qarr[1]={(VkQueue)fakeQueue}; uint32_t qf[1]={0}; qi.numQueues=1; qi.pQueues=qarr; qi.pQueueFamilyIndices=qf;
    ad.header.pNext=&qi.header;
    if(anf.SetInstanceClientAPIAdapterInfo) CK(anf.SetInstanceClientAPIAdapterInfo(inst,&ad.header));

    // create SR technique
    uint32_t IW=1280, IH=720;
    AnfTechniqueVulkanCreateInfo vci; memset(&vci,0,sizeof vci); vci.header.type=ANF_STYPE_TECHNIQUE_VULKAN_CREATE_INFO;
    AnfSRCreateInfo sci; memset(&sci,0,sizeof sci); sci.header.type=ANF_STYPE_SR_CREATE_INFO;
    sci.inputSize.width=IW; sci.inputSize.height=IH; sci.outputSize.width=IW*2; sci.outputSize.height=IH*2;
    AnfTechniqueCreateInfo tci; memset(&tci,0,sizeof tci); tci.header.type=ANF_STYPE_TECHNIQUE_CREATE_INFO;
    tci.techniqueId=tid; tci.flags=0; tci.maxInFlight=2;
    tci.pClientApiCreateInfo=&vci.header; tci.pTechniqueGroupCreateInfo=&sci.header;
    AnfTechnique tech=0;
    if(anf.CreateTechnique) CK(anf.CreateTechnique(inst,&tci,&tech));

    // dispatch SR with dummy resources (handles are fake VkImages; the vk shim serves their info)
    if(tech && anf.DispatchTechnique){
        AnfResourceParamDesc res[4]; memset(res,0,sizeof res);
        struct { AnfResourceLabel lbl; AnfFormat fmt; uint32_t w,h; } R[4]={
            {ANF_RESOURCE_LABEL_INPUT_COLOR, ANF_FORMAT_B10G11R11_UFLOAT, IW,IH},
            {ANF_RESOURCE_LABEL_DEPTH, ANF_FORMAT_D32_FLOAT, IW,IH},
            {ANF_RESOURCE_LABEL_MOTION_VECTORS, ANF_FORMAT_R16G16_FLOAT, IW,IH},
            {ANF_RESOURCE_LABEL_OUTPUT_COLOR, ANF_FORMAT_R16G16B16A16_FLOAT, IW*2,IH*2}};
        for(int i=0;i<4;++i){ res[i].header.type=ANF_STYPE_RESOURCE_PARAM_DESC; res[i].resourceLabel=R[i].lbl;
            res[i].resourceDesc.header.type=ANF_STYPE_RESOURCE_DESC; res[i].resourceDesc.type=ANF_RESOURCE_TYPE_TEXTURE2D;
            res[i].resourceDesc.format=R[i].fmt; res[i].resourceDesc.layout=0;
            res[i].resourceDesc.textureDims.width=R[i].w; res[i].resourceDesc.textureDims.height=R[i].h;
            res[i].resourceDesc.textureDims.mipCount=1; res[i].resourceDesc.resource=(AnfClientResource)(uintptr_t)(0x5000+i); }
        AnfSRDispatch srd; memset(&srd,0,sizeof srd); srd.header.type=ANF_STYPE_SR_DISPATCH; srd.jitterOffset.x=0.1f; srd.jitterOffset.y=-0.2f;
        AnfTechniqueDispatchImmediateInfo imm; memset(&imm,0,sizeof imm); imm.header.type=ANF_STYPE_TECHNIQUE_DISPATCH_IMMEDIATE_INFO;
        AnfTechniqueDispatchInfo di; memset(&di,0,sizeof di); di.header.type=ANF_STYPE_TECHNIQUE_DISPATCH_INFO;
        di.header.pNext=0; di.commandList=(AnfClientCommandList)(uintptr_t)0x6001; di.pResources=res; di.numResources=4; di.reset=ANF_TRUE; di.pTechniqueGroupDispatchInfo=&srd.header;
        for(int frame=0; frame<3; ++frame){ di.reset = frame==0?ANF_TRUE:ANF_FALSE; fprintf(stderr,"--- dispatch frame %d ---\n",frame); CK(anf.DispatchTechnique(tech,&di)); }
    }
    fprintf(stderr,"harness done\n");
    return 0;
}
