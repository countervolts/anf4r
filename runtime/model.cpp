// anf4r: quantized conv network runner. GPL-3.0-only
#include "model.h"
#include <fstream>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <dirent.h>

namespace anf4r {

static std::vector<int8_t> readI8(const std::string& p){std::ifstream f(p,std::ios::binary|std::ios::ate);if(!f)return{};size_t n=f.tellg();f.seekg(0);std::vector<int8_t> v(n);f.read((char*)v.data(),n);return v;}
static std::vector<int32_t> readI32(const std::string& p){std::ifstream f(p,std::ios::binary|std::ios::ate);if(!f)return{};size_t n=f.tellg();f.seekg(0);std::vector<int32_t> v(n/4);f.read((char*)v.data(),n);return v;}
static std::string findFile(const std::string& dir,const char* sub,const char* ext){
    DIR* d=opendir(dir.c_str()); if(!d)return"";std::string r;dirent* e;
    while((e=readdir(d))){std::string s=e->d_name; if(s.find(sub)!=std::string::npos&&s.find(ext)!=std::string::npos){r=dir+"/"+s;break;}}
    closedir(d);return r;
}
static std::vector<float> parseScales(const std::string& p){
    std::ifstream f(p);std::string j((std::istreambuf_iterator<char>(f)),{});std::vector<float> s;
    auto k=j.find("weights_scales");if(k==std::string::npos)return s;auto a=j.find('[',k),e=j.find(']',a);
    std::string arr=j.substr(a+1,e-a-1);size_t i=0;while(i<arr.size()){auto c=arr.find(',',i);std::string t=arr.substr(i,c==std::string::npos?std::string::npos:c-i);
        try{s.push_back(std::stof(t));}catch(...){}if(c==std::string::npos)break;i=c+1;}return s;
}

bool Model::loadSR(const std::string& md,int jp,std::string& err){
    struct L{const char* name;int ic,oc,stride;};
    // QMFNet V2 (prod-7p1): cnn1(20->32 s2) cnn2-0 cnn2-1 (32->32 s1) cnn3(32->64 s1)
    std::vector<L> defs={{"cnn1",20,32,2},{"cnn2-0",32,32,1},{"cnn2-1",32,32,1},{"cnn3",32,64,1}};
    inChannels=20; layers.clear();
    for(auto& d:defs){
        std::string dir=md+"/"+d.name+"/"+std::to_string(jp);
        ConvLayer c; c.name=d.name; c.ic=d.ic; c.oc=d.oc; c.stride=d.stride; c.pad=1; c.relu=true;
        c.weights=readI8(findFile(dir,"weight",".raw"));
        c.bias=readI32(findFile(dir,"bias",".raw"));
        c.scale=parseScales(findFile(dir,"scales",".json"));
        if((int)c.weights.size()!=d.oc*d.ic*9){err="weights "+dir+" got "+std::to_string(c.weights.size());return false;}
        if((int)c.bias.size()!=d.oc){err="bias "+dir;return false;}
        if((int)c.scale.size()!=d.oc){err="scale "+dir+" got "+std::to_string(c.scale.size());return false;}
        layers.push_back(std::move(c));
    }
    return true;
}

static int8_t satTrunc(float f){f=std::trunc(f);if(f<-128)f=-128;if(f>127)f=127;return(int8_t)(int)f;}

static void conv1(const std::vector<int8_t>& in,int IW,int IH,const ConvLayer& L,std::vector<int8_t>& out,int& OW,int& OH){
    OW=(IW+2*L.pad-3)/L.stride+1; OH=(IH+2*L.pad-3)/L.stride+1; out.assign((size_t)OW*OH*L.oc,0);
    for(int oy=0;oy<OH;++oy)for(int ox=0;ox<OW;++ox)for(int oc=0;oc<L.oc;++oc){int acc=0;
        for(int ky=0;ky<3;++ky){int iy=oy*L.stride+ky-L.pad;if(iy<0||iy>=IH)continue;
          for(int kx=0;kx<3;++kx){int ix=ox*L.stride+kx-L.pad;if(ix<0||ix>=IW)continue;
            int inB=(iy*IW+ix)*L.ic,wB=((oc*L.ic)*3+ky)*3+kx;
            for(int ic=0;ic<L.ic;++ic)acc+=int(in[inB+ic])*int(L.weights[wB+ic*9]);}}
        float f=float(acc+L.bias[oc])*L.scale[oc];if(L.relu)f=f<0?0:f;out[(oy*OW+ox)*L.oc+oc]=satTrunc(f);}
}

std::vector<int8_t> runModelCPU(const Model& m,const std::vector<int8_t>& in,int W,int H,int& outW,int& outH){
    std::vector<int8_t> cur=in;int cw=W,ch=H;
    for(auto& L:m.layers){std::vector<int8_t> nx;int ow,oh;conv1(cur,cw,ch,L,nx,ow,oh);cur=std::move(nx);cw=ow;ch=oh;}
    outW=cw;outH=ch;return cur;
}

bool ModelRunner::build(Context& ctx,const Model& m,const std::vector<uint32_t>& spv){
    model_=&m; passes_.resize(m.layers.size());
    wbuf_.resize(m.layers.size());bbuf_.resize(m.layers.size());sbuf_.resize(m.layers.size());
    for(size_t i=0;i<m.layers.size();++i){auto& L=m.layers[i];
        ctx.createBuffer(wbuf_[i],L.weights.size(),true);memcpy(wbuf_[i].map,L.weights.data(),L.weights.size());
        ctx.createBuffer(bbuf_[i],L.bias.size()*4,true);memcpy(bbuf_[i].map,L.bias.data(),L.bias.size()*4);
        ctx.createBuffer(sbuf_[i],L.scale.size()*4,true);memcpy(sbuf_[i].map,L.scale.data(),L.scale.size()*4);
    }
    return true;
}

bool ModelRunner::run(Context& ctx,VkCommandBuffer cb,VkBuffer in,int W,int H,VkBuffer& outBuf,int& outW,int& outH){
    // allocate activation buffers sized to each layer's output
    act_.clear(); passes_.clear(); passes_.resize(model_->layers.size());
    int cw=W,ch=H; VkBuffer curBuf=in;
    for(size_t i=0;i<model_->layers.size();++i){auto& L=model_->layers[i];
        int ow=(cw+2*L.pad-3)/L.stride+1, oh=(ch+2*L.pad-3)/L.stride+1;
        Buffer ob; ctx.createBuffer(ob,(size_t)ow*oh*L.oc,false); act_.push_back(ob);
        std::vector<BindDesc> b={{0,BindKind::StorageBuffer,"in"},{1,BindKind::StorageBuffer,"w"},{2,BindKind::StorageBuffer,"b"},{3,BindKind::StorageBuffer,"s"},{4,BindKind::StorageBuffer,"o"}};
        std::vector<SpecConst> sc={{0,cw},{1,ch},{2,L.ic},{3,L.oc},{4,L.stride},{5,L.pad},{6,L.relu?1:0},{7,ow},{8,oh}};
        auto code=readSpirv("build/conv_int8.spv");
        if(!passes_[i].build(ctx,code,b,sc,0,8,8,1,"main"))return false;
        passes_[i].bindBuffer(0,curBuf);passes_[i].bindBuffer(1,wbuf_[i].buffer);passes_[i].bindBuffer(2,bbuf_[i].buffer);
        passes_[i].bindBuffer(3,sbuf_[i].buffer);passes_[i].bindBuffer(4,ob.buffer);
        passes_[i].record(ctx,cb,ow,oh,L.oc,nullptr,0);
        ctx.barrierAll(cb);
        curBuf=ob.buffer; cw=ow; ch=oh;
    }
    outBuf=curBuf; outW=cw; outH=ch; return true;
}

void ModelRunner::destroyScratch(Context& ctx){for(auto& b:act_)ctx.destroyBuffer(b);act_.clear();for(auto& p:passes_)p.destroy(ctx);passes_.clear();}
void ModelRunner::destroy(Context& ctx){destroyScratch(ctx);for(auto& b:wbuf_)ctx.destroyBuffer(b);for(auto& b:bbuf_)ctx.destroyBuffer(b);for(auto& b:sbuf_)ctx.destroyBuffer(b);}

} // namespace anf4r
