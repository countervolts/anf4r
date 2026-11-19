// anf4r recording libOpenCL.so: enough of the OpenCL + QCOM contract for libanf to set up its SR/FG pipeline,
// logging every call that reveals the kernel sources, build defines, argument layout and dispatch sizes.
// Spoofs an Adreno device so IsTechniqueSupported passes. Returns success + opaque fake handles; nothing executes.
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "log.h"
typedef int32_t cl_int; typedef uint32_t cl_uint; typedef uint64_t cl_ulong; typedef intptr_t cl_properties;
typedef void* cl_platform_id; typedef void* cl_device_id; typedef void* cl_context; typedef void* cl_command_queue;
typedef void* cl_mem; typedef void* cl_program; typedef void* cl_kernel; typedef void* cl_event; typedef void* cl_sampler;
typedef intptr_t cl_device_info; typedef intptr_t cl_program_build_info; typedef intptr_t cl_mem_flags; typedef size_t cl_size;
#define CL_SUCCESS 0
#define FAKE(x) ((void*)(uintptr_t)(0xA000 + (x)))
static int g_kernelCtr=1, g_memCtr=1, g_progCtr=1;

// --- fixed-value accessors ANF uses to gate support ---
cl_int clGetPlatformIDs(cl_uint n, cl_platform_id* ids, cl_uint* num){ LOG("clGetPlatformIDs"); if(num)*num=1; if(ids&&n)ids[0]=FAKE(1); return CL_SUCCESS; }
cl_int clGetPlatformInfo(cl_platform_id p, cl_uint pn, cl_size sz, void* v, cl_size* r){
    const char* s = (pn==0x0902)?"QUALCOMM Snapdragon":(pn==0x0901)?"OpenCL 3.0 Adreno":"cl_qcom_recordable_queues cl_khr_fp16 cl_qcom_inline_asm cl_qcom_extra_vector_types cl_qcom_subgroup_uniform_load cl_qcom_filter_bicubic cl_khr_external_memory cl_khr_external_semaphore cl_qcom_ml_ops";
    if(v&&sz)strncpy(v,s,sz); if(r)*r=strlen(s)+1; LOG("clGetPlatformInfo %#x -> %s",pn,s); return CL_SUCCESS; }
cl_int clGetDeviceIDs(cl_platform_id p, cl_ulong t, cl_uint n, cl_device_id* d, cl_uint* num){ LOG("clGetDeviceIDs"); if(num)*num=1; if(d&&n)d[0]=FAKE(2); return CL_SUCCESS; }
cl_int clGetDeviceInfo(cl_device_id dev, cl_uint pn, cl_size sz, void* v, cl_size* r){
    // Report an Adreno 840 with the QCOM extensions and generous limits.
    if(pn==0x102B){ const char* s="Adreno (TM) 840"; if(v&&sz)strncpy(v,s,sz); if(r)*r=strlen(s)+1; LOG("clGetDeviceInfo NAME"); return CL_SUCCESS; }
    if(pn==0x1030){ const char* s="cl_qcom_recordable_queues cl_khr_fp16 cl_qcom_inline_asm cl_qcom_extra_vector_types cl_qcom_subgroup_uniform_load cl_qcom_filter_bicubic cl_khr_external_memory cl_khr_external_semaphore cl_qcom_ml_ops cl_qcom_accelerated_image_ops"; if(v&&sz)strncpy(v,s,sz); if(r)*r=strlen(s)+1; LOG("clGetDeviceInfo EXTENSIONS"); return CL_SUCCESS; }
    // sane per-query values (avoid huge allocations)
    switch(pn){
      case 0x1000: if(v&&sz>=8)*(uint64_t*)v=4; if(r)*r=8; LOG("clGetDeviceInfo TYPE=GPU"); return CL_SUCCESS; // DEVICE_TYPE=GPU
      case 0x1002: if(v&&sz>=4)*(uint32_t*)v=8;  if(r)*r=4; return CL_SUCCESS; // MAX_COMPUTE_UNITS
      case 0x1003: if(v&&sz>=4)*(uint32_t*)v=3;  if(r)*r=4; return CL_SUCCESS; // MAX_WORK_ITEM_DIMENSIONS
      case 0x1004: if(v&&sz>=8)*(cl_size*)v=1024; if(r)*r=sizeof(cl_size); return CL_SUCCESS; // MAX_WORK_GROUP_SIZE
      case 0x1005: if(!v){if(r)*r=3*sizeof(cl_size);return CL_SUCCESS;} { cl_size* w=(cl_size*)v; for(int i=0;i<3&&(i+1)*sizeof(cl_size)<=sz;i++)w[i]=1024; if(r)*r=3*sizeof(cl_size);} return CL_SUCCESS; // MAX_WORK_ITEM_SIZES
      case 0x100D: if(v&&sz>=8)*(cl_size*)v=4096; if(r)*r=sizeof(cl_size); return CL_SUCCESS; // MAX_PARAMETER_SIZE
      case 0x1010: if(v&&sz>=8)*(uint64_t*)v=1ull<<30; if(r)*r=8; return CL_SUCCESS; // MAX_MEM_ALLOC_SIZE
      case 0x101F: if(v&&sz>=8)*(uint64_t*)v=1ull<<31; if(r)*r=8; return CL_SUCCESS; // GLOBAL_MEM_SIZE
      case 0x1023: if(v&&sz>=8)*(uint64_t*)v=32768; if(r)*r=8; return CL_SUCCESS; // LOCAL_MEM_SIZE
      case 0x100A: if(v&&sz>=4)*(uint32_t*)v=1; if(r)*r=4; return CL_SUCCESS; // IMAGE_SUPPORT=true
      case 0x100B: case 0x100C: case 0x1011: case 0x1012: case 0x1013: case 0x1014: case 0x1015: case 0x1016:
        if(v&&sz>=8)*(cl_size*)v=16384; if(r)*r=sizeof(cl_size); return CL_SUCCESS; // image dims
      case 0x100E: if(v&&sz>=4)*(uint32_t*)v=16; if(r)*r=4; return CL_SUCCESS; // MAX_SAMPLERS
      case 0x1027: if(v&&sz>=4)*(uint32_t*)v=1; if(r)*r=4; return CL_SUCCESS; // CL_DEVICE_AVAILABLE
      case 0x1031: if(v&&sz>=sizeof(void*))*(void**)v=FAKE(1); if(r)*r=sizeof(void*); return CL_SUCCESS; // CL_DEVICE_PLATFORM
      case 0x1042: if(v&&sz>=sizeof(void*))*(void**)v=0; if(r)*r=sizeof(void*); return CL_SUCCESS; // CL_DEVICE_PARENT_DEVICE=NULL
      case 0x1019: if(v&&sz>=4)*(uint32_t*)v=256; if(r)*r=4; return CL_SUCCESS; // MEM_BASE_ADDR_ALIGN
      case 0x1028: if(v&&sz>=4)*(uint32_t*)v=1; if(r)*r=4; return CL_SUCCESS; // COMPILER_AVAILABLE
      case 0x104A: if(v&&sz>=4)*(uint32_t*)v=64; if(r)*r=4; return CL_SUCCESS; // PREFERRED_WG_SIZE_MULTIPLE-ish
    }
    if(pn==0x102c){ const char* q="QUALCOMM"; if(v&&sz)strncpy((char*)v,q,sz); if(r)*r=strlen(q)+1; return CL_SUCCESS; } // VENDOR
    // string queries (names/versions): return a short string
    if(pn==0x102D||pn==0x102E||pn==0x102F||pn==0x103D){ const char* sstr="1.0"; if(v&&sz)strncpy((char*)v,sstr,sz); if(r)*r=strlen(sstr)+1; return CL_SUCCESS; }
    // default: small scalar
    if(v&&sz){ if(sz==4)*(uint32_t*)v=1; else if(sz==8)*(uint64_t*)v=1; else memset(v,0,sz); }
    if(r)*r=sz?sz:4; LOG("clGetDeviceInfo %#x (default)",pn); return CL_SUCCESS; }

// ---- QCOM/KHR extension functions ANF resolves via clGetExtensionFunctionAddressForPlatform ----
cl_int clImportMemoryARM(cl_context c, cl_ulong flags, const void* props, void* fd, cl_size sz, cl_int* e){ LOG("clImportMemoryARM sz=%zu",sz); if(e)*e=CL_SUCCESS; return CL_SUCCESS; }
void* clImportMemoryARM_mem(cl_context c, cl_ulong flags, const void* props, void* fd, cl_size sz, cl_int* e){ if(e)*e=CL_SUCCESS; return FAKE(0x500+g_memCtr++); }
void* clCreateSemaphoreWithPropertiesKHR(cl_context c, const cl_properties* p, cl_int* e){ LOG("clCreateSemaphoreWithPropertiesKHR"); if(e)*e=CL_SUCCESS; return FAKE(0x600); }
cl_int clEnqueueWaitSemaphoresKHR(cl_command_queue q, cl_uint n, const void* s, const cl_ulong* v, cl_uint nw, const cl_event* w, cl_event* ev){ LOG("clEnqueueWaitSemaphoresKHR n=%u",n); if(ev)*ev=FAKE(9); return CL_SUCCESS; }
cl_int clEnqueueSignalSemaphoresKHR(cl_command_queue q, cl_uint n, const void* s, const cl_ulong* v, cl_uint nw, const cl_event* w, cl_event* ev){ LOG("clEnqueueSignalSemaphoresKHR n=%u",n); if(ev)*ev=FAKE(9); return CL_SUCCESS; }
cl_int clGetSemaphoreHandleForTypeKHR(void* s, cl_device_id d, cl_uint t, cl_size sz, void* h, cl_size* r){ if(h&&sz)memset(h,0,sz); if(r)*r=sz; return CL_SUCCESS; }
cl_int clReleaseSemaphoreKHR(void* s){ return CL_SUCCESS; }
cl_int clEnqueueAcquireExternalMemObjectsKHR(cl_command_queue q, cl_uint n, const cl_mem* m, cl_uint nw, const cl_event* w, cl_event* ev){ if(ev)*ev=FAKE(9); return CL_SUCCESS; }
cl_int clEnqueueReleaseExternalMemObjectsKHR(cl_command_queue q, cl_uint n, const cl_mem* m, cl_uint nw, const cl_event* w, cl_event* ev){ if(ev)*ev=FAKE(9); return CL_SUCCESS; }
cl_int clGetKernelSubGroupInfoKHR_ext(cl_kernel k, cl_device_id d, cl_uint pn, cl_size isz, const void* iv, cl_size sz, void* v, cl_size* r){ if(v&&sz){if(sz==8)*(uint64_t*)v=64;else memset(v,0,sz);} if(r)*r=sz; return CL_SUCCESS; }
void* clNewRecordingQCOM(cl_command_queue q, cl_int* e){ LOG("clNewRecordingQCOM"); if(e)*e=CL_SUCCESS; return FAKE(0x700); }
cl_int clUpdateRecordingQCOM(){ LOG("clUpdateRecordingQCOM"); return CL_SUCCESS; }
cl_int clEndRecordingQCOM(void* rec){ LOG("clEndRecordingQCOM"); return CL_SUCCESS; }
cl_int clEnqueueRecordingQCOM(cl_command_queue q, void* rec, cl_uint na, const void* a, cl_uint nw, const cl_event* w, cl_event* ev){ LOG("clEnqueueRecordingQCOM"); if(ev)*ev=FAKE(9); return CL_SUCCESS; }
cl_int clReleaseRecordingQCOM(void* rec){ return CL_SUCCESS; }
cl_int clGetCommandBufferInfoQCOM(void* rec, cl_uint pn, cl_size sz, void* v, cl_size* r){ if(v&&sz)memset(v,0,sz); if(r)*r=sz; return CL_SUCCESS; }
cl_int clEnqueueNDRangeKernelWithPropertiesQCOM(cl_command_queue q, const cl_properties* props, cl_kernel k, cl_uint dim, const cl_size* off, const cl_size* gws, const cl_size* lws, cl_uint nw, const cl_event* w, cl_event* ev){
    char g[64]="",l[64]=""; for(cl_uint i=0;i<dim;++i){ char t[24]; snprintf(t,sizeof t,"%zu,",gws?gws[i]:0); strcat(g,t); if(lws){snprintf(t,sizeof t,"%zu,",lws[i]); strcat(l,t);} }
    LOG("clEnqueueNDRangeKernelWithPropertiesQCOM k=%p dim=%u gws=[%s] lws=[%s]", k, dim, g, l); if(ev)*ev=FAKE(9); return CL_SUCCESS; }
struct ExtE { const char* n; void* f; };
void* clGetExtensionFunctionAddressForPlatform(cl_platform_id p, const char* n){
    static const struct ExtE t[]={
      {"clImportMemoryARM",(void*)clImportMemoryARM_mem},
      {"clCreateSemaphoreWithPropertiesKHR",(void*)clCreateSemaphoreWithPropertiesKHR},
      {"clEnqueueWaitSemaphoresKHR",(void*)clEnqueueWaitSemaphoresKHR},
      {"clEnqueueSignalSemaphoresKHR",(void*)clEnqueueSignalSemaphoresKHR},
      {"clGetSemaphoreHandleForTypeKHR",(void*)clGetSemaphoreHandleForTypeKHR},
      {"clReleaseSemaphoreKHR",(void*)clReleaseSemaphoreKHR},
      {"clEnqueueAcquireExternalMemObjectsKHR",(void*)clEnqueueAcquireExternalMemObjectsKHR},
      {"clEnqueueReleaseExternalMemObjectsKHR",(void*)clEnqueueReleaseExternalMemObjectsKHR},
      {"clGetKernelSubGroupInfoKHR",(void*)clGetKernelSubGroupInfoKHR_ext},
      {"clNewRecordingQCOM",(void*)clNewRecordingQCOM},
      {"clUpdateRecordingQCOM",(void*)clUpdateRecordingQCOM},
      {"clEndRecordingQCOM",(void*)clEndRecordingQCOM},
      {"clEnqueueRecordingQCOM",(void*)clEnqueueRecordingQCOM},
      {"clReleaseRecordingQCOM",(void*)clReleaseRecordingQCOM},
      {"clGetCommandBufferInfoQCOM",(void*)clGetCommandBufferInfoQCOM},
      {"clEnqueueNDRangeKernelWithPropertiesQCOM",(void*)clEnqueueNDRangeKernelWithPropertiesQCOM},
    };
    for(unsigned i=0;i<sizeof t/sizeof*t;i++) if(!strcmp(t[i].n,n)){ LOG("clGetExtFn %s -> impl",n); return t[i].f; }
    LOG("clGetExtFn %s -> NULL",n); return 0; }

cl_context clCreateContext(const cl_properties* props, cl_uint n, const cl_device_id* d, void* cb, void* u, cl_int* e){ LOG("clCreateContext"); if(e)*e=CL_SUCCESS; return FAKE(3); }
cl_command_queue clCreateCommandQueue(cl_context c, cl_device_id d, cl_ulong props, cl_int* e){ LOG("clCreateCommandQueue props=%llu",(unsigned long long)props); if(e)*e=CL_SUCCESS; return FAKE(4); }
cl_command_queue clCreateCommandQueueWithProperties(cl_context c, cl_device_id d, const cl_properties* p, cl_int* e){ LOG("clCreateCommandQueueWithProperties"); if(e)*e=CL_SUCCESS; return FAKE(4); }

cl_program clCreateProgramWithSource(cl_context c, cl_uint n, const char** strs, const cl_size* lens, cl_int* e){
    LOG("clCreateProgramWithSource n=%u (len0=%zu)",n, (n&&lens)?lens[0]:0);
    // dump the source if ANF_TRACE_DUMP set
    const char* dd=getenv("ANF_TRACE_DUMP");
    if(dd && n && strs){ char path[512]; snprintf(path,sizeof path,"%s/src_%d.cl",dd,g_progCtr); FILE* f=fopen(path,"w"); if(f){ for(cl_uint i=0;i<n;++i) fwrite(strs[i],1,lens&&lens[i]?lens[i]:strlen(strs[i]),f); fclose(f);} }
    if(e)*e=CL_SUCCESS; return FAKE(0x100+g_progCtr++); }
cl_program clCreateProgramWithBinary(cl_context c, cl_uint n, const cl_device_id* d, const cl_size* lens, const unsigned char** bins, cl_int* bs, cl_int* e){
    LOG("clCreateProgramWithBinary n=%u len0=%zu",n,(n&&lens)?lens[0]:0); if(bs)bs[0]=CL_SUCCESS; if(e)*e=CL_SUCCESS; return FAKE(0x100+g_progCtr++); }
cl_int clBuildProgram(cl_program p, cl_uint n, const cl_device_id* d, const char* opts, void* cb, void* u){ LOG("clBuildProgram opts=[%s]", opts?opts:""); return CL_SUCCESS; }
cl_int clGetProgramBuildInfo(cl_program p, cl_device_id d, cl_uint pn, cl_size sz, void* v, cl_size* r){ if(v&&sz){ if(pn==0x1183){*(cl_int*)v=0;} else ((char*)v)[0]=0; } if(r)*r=sz?1:0; return CL_SUCCESS; }
cl_int clGetProgramInfo(cl_program p, cl_uint pn, cl_size sz, void* v, cl_size* r){ if(v&&sz)memset(v,0,sz); if(r)*r=sz; return CL_SUCCESS; }

cl_kernel clCreateKernel(cl_program p, const char* name, cl_int* e){ LOG("clCreateKernel %s", name?name:"?"); if(e)*e=CL_SUCCESS; return FAKE(0x200+g_kernelCtr++); }
cl_int clSetKernelArg(cl_kernel k, cl_uint idx, cl_size sz, const void* val){
    // log the argument: size, and the scalar value if small (reveals strides/scales/offsets)
    unsigned long long scal=0; if(val&&sz<=8) memcpy(&scal,val,sz);
    LOG("  clSetKernelArg k=%p idx=%u sz=%zu val=0x%llx (f32=%g)", k, idx, sz, scal, (sz==4)? *(float*)&scal : 0.0);
    return CL_SUCCESS; }
cl_int clGetKernelWorkGroupInfo(cl_kernel k, cl_device_id d, cl_uint pn, cl_size sz, void* v, cl_size* r){ if(v&&sz){ if(sz==8)*(uint64_t*)v=256; else memset(v,0,sz);} if(r)*r=sz; return CL_SUCCESS; }
cl_int clGetKernelSubGroupInfo(cl_kernel k, cl_device_id d, cl_uint pn, cl_size isz, const void* iv, cl_size sz, void* v, cl_size* r){ if(v&&sz){if(sz==8)*(uint64_t*)v=64;else memset(v,0,sz);} if(r)*r=sz; return CL_SUCCESS; }
cl_int clGetKernelSubGroupInfoKHR(cl_kernel k, cl_device_id d, cl_uint pn, cl_size isz, const void* iv, cl_size sz, void* v, cl_size* r){ return clGetKernelSubGroupInfo(k,d,pn,isz,iv,sz,v,r); }

cl_mem clCreateBuffer(cl_context c, cl_mem_flags f, cl_size sz, void* host, cl_int* e){ LOG("clCreateBuffer size=%zu flags=0x%lx", sz, (long)f); if(e)*e=CL_SUCCESS; return FAKE(0x300+g_memCtr++); }
cl_mem clCreateBufferWithProperties(cl_context c, const cl_properties* p, cl_mem_flags f, cl_size sz, void* host, cl_int* e){ LOG("clCreateBufferWithProperties size=%zu",sz); if(e)*e=CL_SUCCESS; return FAKE(0x300+g_memCtr++); }
cl_mem clCreateImage(cl_context c, cl_mem_flags f, const void* fmt, const void* desc, void* host, cl_int* e){ LOG("clCreateImage"); if(e)*e=CL_SUCCESS; return FAKE(0x400+g_memCtr++); }
cl_mem clCreateImage2D(cl_context c, cl_mem_flags f, const void* fmt, cl_size w, cl_size h, cl_size rp, void* host, cl_int* e){ LOG("clCreateImage2D %zux%zu",w,h); if(e)*e=CL_SUCCESS; return FAKE(0x400+g_memCtr++); }
cl_mem clCreateImageWithProperties(cl_context c, const cl_properties* p, cl_mem_flags f, const void* fmt, const void* desc, void* host, cl_int* e){ LOG("clCreateImageWithProperties"); if(e)*e=CL_SUCCESS; return FAKE(0x400+g_memCtr++); }
cl_int clEnqueueNDRangeKernel(cl_command_queue q, cl_kernel k, cl_uint dim, const cl_size* off, const cl_size* gws, const cl_size* lws, cl_uint n, const cl_event* w, cl_event* ev){
    char g[64]="",l[64]=""; for(cl_uint i=0;i<dim;++i){ char t[24]; snprintf(t,sizeof t,"%zu,",gws?gws[i]:0); strcat(g,t); if(lws){snprintf(t,sizeof t,"%zu,",lws[i]); strcat(l,t);} }
    LOG("clEnqueueNDRangeKernel k=%p dim=%u gws=[%s] lws=[%s]", k, dim, g, l); if(ev)*ev=FAKE(9); return CL_SUCCESS; }
cl_int clEnqueueWriteBuffer(cl_command_queue q, cl_mem m, cl_uint block, cl_size off, cl_size sz, const void* p, cl_uint n, const cl_event* w, cl_event* ev){ LOG("clEnqueueWriteBuffer mem=%p off=%zu size=%zu",m,off,sz); if(ev)*ev=FAKE(9); return CL_SUCCESS; }
cl_int clEnqueueReadBuffer(cl_command_queue q, cl_mem m, cl_uint block, cl_size off, cl_size sz, void* p, cl_uint n, const cl_event* w, cl_event* ev){ LOG("clEnqueueReadBuffer mem=%p size=%zu",m,sz); if(p&&sz)memset(p,0,sz); if(ev)*ev=FAKE(9); return CL_SUCCESS; }
// Generic success stubs for the rest
#define STUB0(name) cl_int name(){ LOG(#name); return CL_SUCCESS; }
STUB0(clFinish) STUB0(clFlush) STUB0(clReleaseKernel) STUB0(clReleaseProgram) STUB0(clReleaseMemObject)
STUB0(clReleaseCommandQueue) STUB0(clReleaseContext) STUB0(clReleaseEvent) STUB0(clRetainMemObject)
STUB0(clWaitForEvents) STUB0(clReleaseDevice) STUB0(clRetainEvent) STUB0(clReleaseSampler)
void* clSVMAlloc(cl_context c, cl_ulong f, cl_size sz, cl_uint a){ LOG("clSVMAlloc %zu",sz); return malloc(sz?sz:1); }
void clSVMFree(cl_context c, void* p){ free(p); }
cl_int clGetEventProfilingInfo(cl_event e, cl_uint pn, cl_size sz, void* v, cl_size* r){ if(v&&sz)memset(v,0,sz); if(r)*r=sz; return CL_SUCCESS; }
cl_int clGetMemObjectInfo(cl_mem m, cl_uint pn, cl_size sz, void* v, cl_size* r){ if(v&&sz)memset(v,0,sz); if(r)*r=sz; return CL_SUCCESS; }
cl_sampler clCreateSampler(cl_context c, cl_uint nc, cl_uint am, cl_uint fm, cl_int* e){ if(e)*e=CL_SUCCESS; return FAKE(7); }
