// anf4r: run the full QMFNet SR network on GPU, validate bit-exact vs CPU chain + determinism.
#include "../runtime/model.h"
#include <cstdio>
#include <cstring>
#include <random>
using namespace anf4r;

int main(int argc,char**argv){
    const char* md=argc>1?argv[1]:"vendor/assets/modelzoo/aiff-sr-v2s-prod-7p1-invpow2";
    int jp=argc>2?atoi(argv[2]):0;
    Model m; std::string err; if(!m.loadSR(md,jp,err)){printf("load: %s\n",err.c_str());return 1;}
    printf("model %s phase %d: %zu layers, in %dch\n",md,jp,m.layers.size(),m.inChannels);
    int W=96,H=64; std::vector<int8_t> in((size_t)W*H*m.inChannels);
    std::mt19937 rng(7); std::uniform_int_distribution<int> d(-128,127); for(auto&x:in)x=(int8_t)d(rng);
    int cw,ch; auto ref=runModelCPU(m,in,W,H,cw,ch);
    printf("cpu out %dx%dx%d = %zu\n",cw,ch,m.layers.back().oc,ref.size());

    Context ctx; if(!ctx.create(getenv("VAL"))){printf("ctx %s\n",ctx.error.c_str());return 1;}
    ModelRunner mr; mr.build(ctx,m,{});
    Buffer bin; ctx.createBuffer(bin,in.size(),true); memcpy(bin.map,in.data(),in.size());
    auto once=[&](std::vector<int8_t>& out){
        VkBuffer ob;int ow,oh; VkCommandBuffer cb=ctx.begin();
        if(!mr.run(ctx,cb,bin.buffer,W,H,ob,ow,oh)){printf("run fail %s\n",ctx.error.c_str());exit(1);} ctx.submit(cb);
        out.resize((size_t)ow*oh*m.layers.back().oc);
        // copy device out buffer to host
        Buffer host;ctx.createBuffer(host,out.size(),true);VkCommandBuffer cb2=ctx.begin();
        VkBufferCopy bc{0,0,out.size()};vkCmdCopyBuffer(cb2,ob,host.buffer,1,&bc);ctx.submit(cb2);
        memcpy(out.data(),host.map,out.size());ctx.destroyBuffer(host);mr.destroyScratch(ctx);
    };
    std::vector<int8_t> g1,g2; once(g1); once(g2);
    size_t d1=0;for(size_t i=0;i<g1.size();++i)if(g1[i]!=ref[i])d1++;
    size_t dd=0;for(size_t i=0;i<g1.size();++i)if(g1[i]!=g2[i])dd++;
    printf("GPU vs CPU mismatches=%zu ; determinism (run1 vs run2) diffs=%zu\n",d1,dd);
    printf((d1==0&&dd==0)?"MODEL OK: full SR net bit-exact to CPU ref and deterministic\n":"MODEL FAIL\n");
    mr.destroy(ctx);ctx.destroyBuffer(bin);ctx.destroy();
    return (d1==0&&dd==0)?0:2;
}
