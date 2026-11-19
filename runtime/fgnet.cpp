// anf4r: FG flow networks. GPL-3.0-only
#include "fgnet.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

namespace anf4r {

// ---------------- file helpers ----------------
static std::vector<uint8_t> readAll(const std::string& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate); if (!f) return {};
    size_t n = f.tellg(); f.seekg(0); std::vector<uint8_t> v(n); f.read((char*)v.data(), n); return v;
}
static std::string readText(const std::string& p) { auto v = readAll(p); return std::string(v.begin(), v.end()); }
// value of a JSON key that is either a number or a flat array of numbers
static std::vector<float> jsonNums(const std::string& j, const char* key) {
    std::vector<float> r; auto k = j.find(std::string("\"") + key + "\""); if (k == std::string::npos) return r;
    k = j.find(':', k) + 1; while (k < j.size() && j[k] == ' ') ++k;
    if (j[k] == '[') { auto e = j.find(']', k); std::stringstream ss(j.substr(k + 1, e - k - 1)); std::string t;
        while (std::getline(ss, t, ',')) r.push_back(std::stof(t)); }
    else r.push_back(std::stof(j.substr(k)));
    return r;
}
static float jsonObjNum(const std::string& j, const char* obj, const char* key) {
    auto o = j.find(std::string("\"") + obj + "\""); auto k = j.find(std::string("\"") + key + "\"", o);
    k = j.find(':', k) + 1; return std::stof(j.substr(k));
}

float halfToFloat(uint16_t h) {
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023; float v;
    if (e == 0) v = std::ldexp((float)m, -24); else if (e == 31) v = m ? NAN : INFINITY; else v = std::ldexp((float)(m + 1024), (int)e - 25);
    return s ? -v : v;
}
uint16_t floatToHalf(float f) {   // RTE
    uint32_t b; memcpy(&b, &f, 4); uint32_t s = (b >> 16) & 0x8000; int e = (int)((b >> 23) & 0xff) - 127 + 15; uint32_t m = b & 0x7fffff;
    if (((b >> 23) & 0xff) == 255) return (uint16_t)(s | 0x7c00 | (m ? 0x200 : 0));
    if (e >= 31) return (uint16_t)(s | 0x7c00);
    if (e <= 0) { if (e < -10) return (uint16_t)s; m |= 0x800000; int sh = 14 - e; uint32_t r = m >> sh, rem = m & ((1u << sh) - 1), half = 1u << (sh - 1);
        if (rem > half || (rem == half && (r & 1))) ++r; return (uint16_t)(s | r); }
    uint32_t r = (uint32_t)(e << 10) | (m >> 13), rem = m & 0x1fff;
    if (rem > 0x1000 || (rem == 0x1000 && (r & 1))) ++r;
    return (uint16_t)(s | r);
}

// ---------------- model loading ----------------
bool FgNetModel::load(const std::string& dir, bool lr, std::string& err) {
    isLR = lr; hidden.clear();
    std::string enc_ = readText(dir + "/network_io_encodings.json");
    if (enc_.empty()) { err = "no network_io_encodings.json"; return false; }
    enc.inScale = jsonObjNum(enc_, "input", "scale"); enc.inOffset = jsonObjNum(enc_, "input", "offset");
    enc.flowScale = jsonObjNum(enc_, "flow", "scale"); enc.flowOffset = jsonObjNum(enc_, "flow", "offset");
    enc.maskScale = jsonObjNum(enc_, "mask", "scale"); enc.maskOffset = jsonObjNum(enc_, "mask", "offset");
    struct D { const char* n; int ic, oc, s; };
    int C = lr ? 96 : 32;
    std::vector<D> defs = {{"conv0_0", 32, C, 2}, {"conv0_1", C, C, 2}};
    const char* names[] = {"conv1_0","conv1_1","conv2_0","conv2_1","conv3_0","conv3_1","conv4_0","conv4_1","conv5_0","conv5_1"};
    for (auto n : names) defs.push_back({n, C, C, 1});
    defs.push_back({"conv_out", C, lr ? 64 : 32, 1});
    auto loadLayer = [&](const D& d, bool fp16w, FgLayer& L) -> bool {
        std::string ld = dir + "/" + d.n + "/jitter0";
        L.name = d.n; L.ic = d.ic; L.oc = d.oc; L.stride = d.s;
        auto w = readAll(ld + "/weights.raw"), b = readAll(ld + "/bias.raw");
        std::string sj = readText(ld + "/scales.json");
        auto ws = jsonNums(sj, "weights_scales"), ias = jsonNums(sj, "input_activation_scale"), oas = jsonNums(sj, "output_activation_scale");
        size_t n = (size_t)d.oc * d.ic * 9;
        if (fp16w) {
            if (w.size() != n * 2 || b.size() != (size_t)d.oc * 2) { err = "fp16 head sizes " + ld; return false; }
            std::vector<uint16_t> src(n); memcpy(src.data(), w.data(), n * 2);
            L.w16.assign(n, 0);
            for (int o = 0; o < d.oc; ++o) for (int c = 0; c < d.ic; ++c) for (int k = 0; k < 9; ++k)
                L.w16[((size_t)o * 9 + k) * d.ic + c] = src[((size_t)o * d.ic + c) * 9 + k];
            L.biasf.resize(d.oc); for (int o = 0; o < d.oc; ++o) { uint16_t h; memcpy(&h, &b[o * 2], 2); L.biasf[o] = halfToFloat(h); }
            return true;
        }
        if (w.size() != n || b.size() != (size_t)d.oc * 4 || (int)ws.size() != d.oc) { err = "int8 layer sizes " + ld; return false; }
        L.w8.assign(n, 0);
        for (int o = 0; o < d.oc; ++o) for (int c = 0; c < d.ic; ++c) for (int k = 0; k < 9; ++k)
            L.w8[((size_t)o * 9 + k) * d.ic + c] = (int8_t)w[((size_t)o * d.ic + c) * 9 + k];
        L.scale.resize(d.oc); L.biasf.resize(d.oc);
        for (int o = 0; o < d.oc; ++o) {
            float ia = ias.size() == 1 ? ias[0] : ias[o], oa = oas.size() == 1 ? oas[0] : oas[o];
            L.scale[o] = ws[o] * ia / oa;
            int32_t bi; memcpy(&bi, &b[o * 4], 4); L.biasf[o] = (float)bi * L.scale[o];
        }
        return true;
    };
    for (auto& d : defs) { FgLayer L; if (!loadLayer(d, false, L)) return false; hidden.push_back(std::move(L)); }
    return loadLayer({"conv_out_flow_mask", lr ? 64 : 32, 20, 1}, lr, head);
}

FgNetDims fgNetDims(int W, int H) {
    int a = (W + 1) / 2, b = (H + 1) / 2; FgNetDims d; d.nw = (a + 1) / 2; d.nh = (b + 1) / 2; d.flowW = d.nw * 2; d.flowH = d.nh * 2; return d;
}

// ---------------- CPU references ----------------
static inline float rndazf(float x) { return x >= 0 ? std::floor(x + 0.5f) : std::ceil(x - 0.5f); }
static int accI8(const FgLayer& L, const std::vector<int8_t>& in, int W, int H, int ox, int oy, int oc) {
    int a = 0;
    for (int ky = 0; ky < 3; ++ky) { int iy = oy * L.stride + ky - 1; if (iy < 0 || iy >= H) continue;
        for (int kx = 0; kx < 3; ++kx) { int ix = ox * L.stride + kx - 1; if (ix < 0 || ix >= W) continue;
            const int8_t* ip = &in[((size_t)iy * W + ix) * L.ic]; const int8_t* wp = &L.w8[(((size_t)oc * 3 + ky) * 3 + kx) * L.ic];
            for (int c = 0; c < L.ic; ++c) a += (int)ip[c] * (int)wp[c]; } }
    return a;
}
std::vector<int8_t> fgConvI8CPU(const FgLayer& L, const std::vector<int8_t>& in, int W, int H, int& OW, int& OH, int lo, int hi) {
    OW = (W - 1) / L.stride + 1; OH = (H - 1) / L.stride + 1; std::vector<int8_t> out((size_t)OW * OH * L.oc);
    for (int y = 0; y < OH; ++y) for (int x = 0; x < OW; ++x) for (int c = 0; c < L.oc; ++c) {
        volatile float f = std::fma(L.scale[c], (float)accI8(L, in, W, H, x, y, c), L.biasf[c]);
        float g = std::min(std::max((float)f, (float)lo), (float)hi); out[((size_t)y * OW + x) * L.oc + c] = (int8_t)(int)rndazf(g); }
    return out;
}
std::vector<uint16_t> fgConvI8ToHalfCPU(const FgLayer& L, const std::vector<int8_t>& in, int W, int H, int& OW, int& OH) {
    OW = (W - 1) / L.stride + 1; OH = (H - 1) / L.stride + 1; std::vector<uint16_t> out((size_t)OW * OH * L.oc);
    for (int y = 0; y < OH; ++y) for (int x = 0; x < OW; ++x) for (int c = 0; c < L.oc; ++c) {
        volatile float f = std::fma(L.scale[c], (float)accI8(L, in, W, H, x, y, c), L.biasf[c]);
        out[((size_t)y * OW + x) * L.oc + c] = floatToHalf(std::max((float)f, 0.0f)); }
    return out;
}
static float headLR(const FgLayer& L, const std::vector<uint16_t>& in, int W, int H, int ox, int oy, int oc) {
    float acc = 0;
    for (int ky = 0; ky < 3; ++ky) { int iy = oy + ky - 1; if (iy < 0 || iy >= H) continue;
        for (int kx = 0; kx < 3; ++kx) { int ix = ox + kx - 1; if (ix < 0 || ix >= W) continue;
            const uint16_t* ip = &in[((size_t)iy * W + ix) * 64]; const uint16_t* wp = &L.w16[(((size_t)oc * 3 + ky) * 3 + kx) * 64];
            for (int c = 0; c < 64; ++c) { volatile float p = halfToFloat(ip[c]) * halfToFloat(wp[c]); acc += p; } } }
    return acc + L.biasf[oc];
}
void fgHeadLRCPU(const FgLayer& L, const std::vector<uint16_t>& in, int W, int H, std::vector<uint16_t>& flow, std::vector<uint16_t>& mask) {
    flow.assign((size_t)(2 * H) * (2 * W) * 4, 0); mask.assign((size_t)((W * H + 1) / 2 * 2), 0);
    for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
        for (int z = 0; z < 4; ++z) { int px = x * 2 + (z & 1), py = y * 2 + (z >> 1);
            for (int k = 0; k < 4; ++k) flow[((size_t)py * 2 * W + px) * 4 + k] = floatToHalf(headLR(L, in, W, H, x, y, z * 4 + k)); }
        mask[(size_t)y * W + x] = floatToHalf(headLR(L, in, W, H, x, y, 16)); }
}
static uint8_t q8(const FgLayer& L, const std::vector<int8_t>& in, int W, int H, int x, int y, int oc, int zp) {
    volatile float f = std::fma(L.scale[oc], (float)accI8(L, in, W, H, x, y, oc), L.biasf[oc]);
    float g = rndazf((float)f) + (float)zp; g = std::min(std::max(g, (float)zp), 255.0f); return (uint8_t)(unsigned)g;
}
void fgHeadQRCPU(const FgLayer& L, const std::vector<int8_t>& in, int W, int H, int flowZp, int maskZp, std::vector<uint8_t>& flow, std::vector<uint8_t>& mask) {
    flow.assign((size_t)(2 * H) * (2 * W) * 4, 0); mask.assign((size_t)W * H, 0);
    FgLayer l = L; l.stride = 1;
    for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
        for (int z = 0; z < 4; ++z) { int px = x * 2 + (z & 1), py = y * 2 + (z >> 1);
            for (int k = 0; k < 4; ++k) flow[((size_t)py * 2 * W + px) * 4 + k] = q8(l, in, W, H, x, y, z * 4 + k, flowZp); }
        mask[(size_t)y * W + x] = q8(l, in, W, H, x, y, 16, maskZp); }
}

// ---------------- GPU ----------------
static void upBuf(Context& c, Buffer& b, const void* d, size_t n) { c.createBuffer(b, n ? n : 4, true); memcpy(b.map, d, n); }

bool FgNet::build(Context& ctx, const FgNetModel& m, int W, int H, const std::string& spvDir) {
    m_ = &m; dims = fgNetDims(W, H);
    auto cs = readSpirv(spvDir + "/fg_conv_i8.spv"); auto hl = readSpirv(spvDir + "/fg_head_lr.spv"); auto hq = readSpirv(spvDir + "/fg_head_qr.spv");
    if (cs.empty() || hl.empty() || hq.empty()) { ctx.error = "missing fg spv in " + spvDir; return false; }
    size_t nl = m.hidden.size();
    wbuf_.resize(nl + 1); bbuf_.resize(nl + 1); sbuf_.resize(nl + 1);
    for (size_t i = 0; i < nl; ++i) { auto& L = m.hidden[i];
        upBuf(ctx, wbuf_[i], L.w8.data(), L.w8.size()); upBuf(ctx, bbuf_[i], L.biasf.data(), L.biasf.size() * 4); upBuf(ctx, sbuf_[i], L.scale.data(), L.scale.size() * 4); }
    if (m.isLR) { upBuf(ctx, wbuf_[nl], m.head.w16.data(), m.head.w16.size() * 2); upBuf(ctx, bbuf_[nl], m.head.biasf.data(), m.head.biasf.size() * 4); }
    else { upBuf(ctx, wbuf_[nl], m.head.w8.data(), m.head.w8.size()); upBuf(ctx, bbuf_[nl], m.head.biasf.data(), m.head.biasf.size() * 4); upBuf(ctx, sbuf_[nl], m.head.scale.data(), m.head.scale.size() * 4); }

    passes_.assign(nl + 2, Pass());   // hidden..., head flow, head mask
    act_.resize(nl);
    int cw = W, ch = H;
    std::vector<int> outW(nl), outH(nl);
    for (size_t i = 0; i < nl; ++i) {
        auto& L = m.hidden[i]; int ow = (cw - 1) / L.stride + 1, oh = (ch - 1) / L.stride + 1;
        bool half = m.isLR && i == nl - 1;
        size_t bytes = (size_t)ow * oh * L.oc * (half ? 2 : 1);
        ctx.createBuffer(act_[i], (bytes + 3) & ~size_t(3), false);
        std::vector<BindDesc> b = {{0,BindKind::StorageBuffer,"in"},{1,BindKind::StorageBuffer,"w"},{2,BindKind::StorageBuffer,"b"},{3,BindKind::StorageBuffer,"s"},{4,BindKind::StorageBuffer,"o"},{5,BindKind::StorageBuffer,"o8"}};
        std::vector<SpecConst> sc = {{0,cw},{1,ch},{2,L.ic},{3,L.oc},{4,L.stride},{5,ow},{6,oh},{7,half?1:0},{8,0},{9,127}};
        if (!passes_[i].build(ctx, cs, b, sc, 0, 8, 8, 1, "main")) return false;
        passes_[i].bindBuffer(1, wbuf_[i].buffer); passes_[i].bindBuffer(2, bbuf_[i].buffer); passes_[i].bindBuffer(3, sbuf_[i].buffer);
        passes_[i].bindBuffer(4, act_[i].buffer); passes_[i].bindBuffer(5, act_[i].buffer);
        int gz = half ? (L.oc + 1) / 2 : L.oc; steps_.push_back({(int)i, ow, oh, gz});
        cw = ow; ch = oh; outW[i] = ow; outH[i] = oh;
    }
    // head
    int nw = cw, nh = ch;
    if (nw != dims.nw || nh != dims.nh) { ctx.error = "dims mismatch"; return false; }
    size_t fb, mb;
    if (m.isLR) { fb = (size_t)dims.flowW * dims.flowH * 8; mb = ((size_t)nw * nh * 2 + 3) & ~size_t(3); }
    else { fb = (size_t)dims.flowW * dims.flowH * 4; mb = ((size_t)nw * nh + 3) & ~size_t(3); }
    ctx.createBuffer(flow, fb, false); ctx.createBuffer(mask, mb, false);
    for (int mode = 0; mode < 2; ++mode) {
        Pass& p = passes_[nl + mode];
        if (m.isLR) {
            std::vector<BindDesc> b = {{0,BindKind::StorageBuffer,"in"},{1,BindKind::StorageBuffer,"w"},{2,BindKind::StorageBuffer,"b"},{3,BindKind::StorageBuffer,"o"}};
            if (!p.build(ctx, hl, b, {{0,nw},{1,nh},{2,mode}}, 0, 8, 8, 1, "main")) return false;
            p.bindBuffer(1, wbuf_[nl].buffer); p.bindBuffer(2, bbuf_[nl].buffer); p.bindBuffer(3, (mode ? mask : flow).buffer);
            steps_.push_back(mode ? Step{(int)(nl + mode), (nw + 1) / 2, nh, 1} : Step{(int)(nl + mode), nw, nh, 4});
        } else {
            std::vector<BindDesc> b = {{0,BindKind::StorageBuffer,"in"},{1,BindKind::StorageBuffer,"w"},{2,BindKind::StorageBuffer,"b"},{3,BindKind::StorageBuffer,"s"},{4,BindKind::StorageBuffer,"o"}};
            int pitch = mode ? nw : dims.flowW;
            if (!p.build(ctx, hq, b, {{0,nw},{1,nh},{2,mode},{3,pitch},{4,(int)m.enc.flowOffset},{5,(int)m.enc.maskOffset}}, 0, 8, 8, 1, "main")) return false;
            p.bindBuffer(1, wbuf_[nl].buffer); p.bindBuffer(2, bbuf_[nl].buffer); p.bindBuffer(3, sbuf_[nl].buffer); p.bindBuffer(4, (mode ? mask : flow).buffer);
            steps_.push_back({(int)(nl + mode), nw, nh, mode ? 1 : 4});
        }
    }
    return true;
}

void FgNet::record(Context& ctx, VkCommandBuffer cb, VkBuffer input) {
    size_t nl = m_->hidden.size(); VkBuffer cur = input;
    for (size_t s = 0; s < steps_.size(); ++s) {
        auto& st = steps_[s]; Pass& p = passes_[st.pass];
        p.bindBuffer(0, cur);
        p.record(ctx, cb, st.gx, st.gy, st.gz, nullptr, 0);
        ctx.barrierAll(cb);
        if (st.pass < (int)nl) cur = act_[st.pass].buffer;   // head passes (flow, mask) both read the last hidden activation
    }
}

void FgNet::destroy(Context& ctx) {
    for (auto& p : passes_) p.destroy(ctx);
    for (auto* v : {&wbuf_, &bbuf_, &sbuf_, &act_}) { for (auto& b : *v) if (b.buffer) ctx.destroyBuffer(b); v->clear(); }
    ctx.destroyBuffer(flow); ctx.destroyBuffer(mask); passes_.clear(); steps_.clear();
}

} // namespace anf4r
