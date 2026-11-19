// anf4r: validate the int8 conv GPU kernel against a CPU reference, bit-exact, on real ANF weights.
#include "../runtime/pass.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <string>
#include <fstream>
#include <random>
using namespace anf4r;

static std::vector<int8_t> readI8(const std::string& p){ std::ifstream f(p,std::ios::binary|std::ios::ate); size_t n=f.tellg(); f.seekg(0); std::vector<int8_t> v(n); f.read((char*)v.data(),n); return v;}
static std::vector<int32_t> readI32(const std::string& p){ std::ifstream f(p,std::ios::binary|std::ios::ate); size_t n=f.tellg(); f.seekg(0); std::vector<int32_t> v(n/4); f.read((char*)v.data(),n); return v;}

static int8_t satTrunc(float f){ f=std::trunc(f); if(f<-128)f=-128; if(f>127)f=127; return (int8_t)(int)f; }

// CPU reference: identical arithmetic to the GLSL kernel.
static void convRef(const std::vector<int8_t>& in,int IW,int IH,int IC,
                    const std::vector<int8_t>& w,const std::vector<int32_t>& bias,const std::vector<float>& scale,
                    int OC,int stride,int pad,bool relu,std::vector<int8_t>& out,int OW,int OH){
    out.assign((size_t)OW*OH*OC,0);
    for(int oy=0;oy<OH;++oy)for(int ox=0;ox<OW;++ox)for(int oc=0;oc<OC;++oc){
        int acc=0;
        for(int ky=0;ky<3;++ky){int iy=oy*stride+ky-pad; if(iy<0||iy>=IH)continue;
          for(int kx=0;kx<3;++kx){int ix=ox*stride+kx-pad; if(ix<0||ix>=IW)continue;
            int inB=(iy*IW+ix)*IC; int wB=((oc*IC)*3+ky)*3+kx;
            for(int ic=0;ic<IC;++ic) acc+=int(in[inB+ic])*int(w[wB+ic*9]);
          }}
        float f=float(acc+bias[oc])*scale[oc]; if(relu) f=f<0?0:f; out[(oy*OW+ox)*OC+oc]=satTrunc(f);
    }
}

int main(int argc,char**argv){
    const char* layerDir = argc>1?argv[1]:"vendor/assets/modelzoo/aiff-sr-v2s-prod-7p1-invpow2/cnn1/0";
    int IC=20,OC=32,stride=2; // cnn1 defaults
    if(argc>2) IC=atoi(argv[2]); if(argc>3) OC=atoi(argv[3]); if(argc>4) stride=atoi(argv[4]);
    // load weights: find the weight/bias/scales files
    std::string wp,bp,sp; { std::string d=layerDir;
      // guess filenames
      FILE* ls=popen(("ls "+d).c_str(),"r"); char buf[512];
      while(fgets(buf,512,ls)){std::string s=buf; s.erase(s.find_last_not_of(" \n\r")+1);
        if(s.find("weight")!=std::string::npos&&s.find(".raw")!=std::string::npos) wp=d+"/"+s;
        if(s.find("bias")!=std::string::npos&&s.find(".raw")!=std::string::npos) bp=d+"/"+s;
        if(s.find("scales")!=std::string::npos) sp=d+"/"+s; } pclose(ls); }
    auto w=readI8(wp); auto bias=readI32(bp);
    // parse scales.json weights_scales array
    std::vector<float> scale; { std::ifstream f(sp); std::string j((std::istreambuf_iterator<char>(f)),{});
      auto p=j.find("weights_scales"); p=j.find('[',p); auto e=j.find(']',p);
      std::string arr=j.substr(p+1,e-p-1); size_t i=0; while(i<arr.size()){ size_t c=arr.find(',',i); std::string tok=arr.substr(i,c==std::string::npos?std::string::npos:c-i);
        scale.push_back(std::stof(tok)); if(c==std::string::npos)break; i=c+1; } }
    printf("layer %s\n  w=%zu (expect %d) bias=%zu scale=%zu  IC=%d OC=%d stride=%d\n",layerDir,w.size(),OC*IC*9,bias.size(),scale.size(),IC,OC,stride);
    if((int)w.size()!=OC*IC*9){printf("WEIGHT SIZE MISMATCH\n");return 1;}

    int IW=64,IH=48; int OW=(IW+2*1-3)/stride+1, OH=(IH+2*1-3)/stride+1;
    std::vector<int8_t> in((size_t)IW*IH*IC);
    std::mt19937 rng(1234); std::uniform_int_distribution<int> d(-128,127);
    for(auto&x:in)x=(int8_t)d(rng);

    std::vector<int8_t> ref; convRef(in,IW,IH,IC,w,bias,scale,OC,stride,1,true,ref,OW,OH);

    // GPU
    Context ctx; if(!ctx.create(getenv("VAL"))){printf("ctx %s\n",ctx.error.c_str());return 1;}
    Buffer bin,bw,bb,bs,bo;
    ctx.createBuffer(bin,in.size(),true); memcpy(bin.map,in.data(),in.size());
    ctx.createBuffer(bw,w.size(),true); memcpy(bw.map,w.data(),w.size());
    ctx.createBuffer(bb,bias.size()*4,true); memcpy(bb.map,bias.data(),bias.size()*4);
    ctx.createBuffer(bs,scale.size()*4,true); memcpy(bs.map,scale.data(),scale.size()*4);
    ctx.createBuffer(bo,(size_t)OW*OH*OC,true);
    auto code=readSpirv("build/conv_int8.spv");
    Pass pass; std::vector<BindDesc> binds={{0,BindKind::StorageBuffer,"in"},{1,BindKind::StorageBuffer,"w"},{2,BindKind::StorageBuffer,"b"},{3,BindKind::StorageBuffer,"s"},{4,BindKind::StorageBuffer,"o"}};
    std::vector<SpecConst> specs={{0,IW},{1,IH},{2,IC},{3,OC},{4,stride},{5,1},{6,1},{7,OW},{8,OH}};
    if(!pass.build(ctx,code,binds,specs,0,8,8,1,"main")){printf("build %s\n",ctx.error.c_str());return 1;}
    pass.bindBuffer(0,bin.buffer);pass.bindBuffer(1,bw.buffer);pass.bindBuffer(2,bb.buffer);pass.bindBuffer(3,bs.buffer);pass.bindBuffer(4,bo.buffer);
    VkCommandBuffer cb=ctx.begin(); pass.record(ctx,cb,OW,OH,OC,nullptr,0); ctx.barrierAll(cb); ctx.submit(cb);
    std::vector<int8_t> gpu((size_t)OW*OH*OC); memcpy(gpu.data(),bo.map,gpu.size());

    size_t diff=0,maxd=0; for(size_t i=0;i<gpu.size();++i){int dd=abs(int(gpu[i])-int(ref[i])); if(dd){diff++; if((size_t)dd>maxd)maxd=dd;}}
    printf("  out %dx%dx%d = %zu values; mismatches=%zu maxdiff=%zu\n",OW,OH,OC,gpu.size(),diff,maxd);
    printf(diff==0?"CONV BIT-EXACT: GPU == CPU reference\n":"CONV MISMATCH\n");
    return diff==0?0:2;
}
