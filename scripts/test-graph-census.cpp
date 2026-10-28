#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama-cpp.h"
#include "../src/llama-ext.h"
#include "ggml.h"
#include "gguf-model-data.h"
#include "gguf.h"
#include "ggml-backend.h"
#include "download.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <vector>
#include <set>
#include <map>
#include <tuple>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <ctime>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <cstring>
#include <filesystem>

// ---------- SHA256 (minimal) ----------
namespace {
struct SHA256 {
    uint32_t state[8];
    uint64_t bitlen;
    uint8_t buffer[64];
    size_t buffer_len;
    SHA256() {
        state[0]=0x6a09e667; state[1]=0xbb67ae85; state[2]=0x3c6ef372; state[3]=0xa54ff53a;
        state[4]=0x510e527f; state[5]=0x9b05688c; state[6]=0x1f83d9ab; state[7]=0x5be0cd19;
        bitlen=0; buffer_len=0;
    }
    static uint32_t rotr(uint32_t x,int n){return (x>>n)|(x<<(32-n));}
    void compress(const uint8_t * data){
        uint32_t w[64];
        for(int i=0;i<16;i++){
            w[i]=((uint32_t)data[i*4]<<24)|((uint32_t)data[i*4+1]<<16)|((uint32_t)data[i*4+2]<<8)|((uint32_t)data[i*4+3]);
        }
        for(int i=16;i<64;i++){
            uint32_t s0=rotr(w[i-15],7)^rotr(w[i-15],18)^(w[i-15]>>3);
            uint32_t s1=rotr(w[i-2],17)^rotr(w[i-2],19)^(w[i-2]>>10);
            w[i]=w[i-16]+s0+w[i-7]+s1;
        }
        uint32_t a=state[0],b=state[1],c=state[2],d=state[3],e=state[4],f=state[5],g=state[6],h=state[7];
        static const uint32_t k[64]={
            0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
            0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
            0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
            0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
            0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
            0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
            0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
            0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
        for(int i=0;i<64;i++){
            uint32_t S1=rotr(e,6)^rotr(e,11)^rotr(e,25);
            uint32_t ch=(e&f)^(~e&g);
            uint32_t temp1=h+S1+ch+k[i]+w[i];
            uint32_t S0=rotr(a,2)^rotr(a,13)^rotr(a,22);
            uint32_t maj=(a&b)^(a&c)^(b&c);
            uint32_t temp2=S0+maj;
            h=g; g=f; f=e; e=d+temp1; d=c; c=b; b=a; a=temp1+temp2;
        }
        state[0]+=a; state[1]+=b; state[2]+=c; state[3]+=d; state[4]+=e; state[5]+=f; state[6]+=g; state[7]+=h;
    }
    void update(const uint8_t * data, size_t len){
        for(size_t i=0;i<len;i++){
            buffer[buffer_len++]=data[i];
            if(buffer_len==64){ compress(buffer); buffer_len=0; }
        }
        bitlen += len*8;
    }
    std::string final(){
        buffer[buffer_len++] = 0x80;
        if (buffer_len > 56) {
            while (buffer_len < 64) buffer[buffer_len++] = 0;
            compress(buffer);
            buffer_len = 0;
        }
        while (buffer_len < 56) buffer[buffer_len++] = 0;
        uint64_t bl = bitlen;
        for (int i = 7; i >= 0; i--) { buffer[buffer_len++] = (uint8_t)(bl >> (i*8)); }
        compress(buffer);
        char out[65];
        for (int i = 0; i < 8; i++) sprintf(out + i * 8, "%08x", state[i]);
        out[64] = 0;
        return std::string(out);
    }
};
std::string sha256_file(const std::string & path){
    std::ifstream f(path, std::ios::binary);
    if(!f) return "";
    SHA256 s;
    char buf[65536];
    while(true){
        f.read(buf, sizeof(buf));
        size_t n = f.gcount();
        if(n>0) s.update((const uint8_t*)buf, n);
        if(!f) break;
    }
    return s.final();
}
std::string sha256_self(){
    char buf[4096];
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf)-1);
    if(len==-1) return "";
    buf[len]=0;
    return sha256_file(std::string(buf));
}
}

static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    GGML_UNUSED(tensor);
    GGML_UNUSED(userdata);
}

static std::string json_escape(const std::string & s){
    std::string out;
    for(char c: s){
        switch(c){
            case '"': out += "\\"; out += "\""; break;
            case '\\': out+="\\\\"; break;
            case '\n': out+="\\n"; break;
            case '\r': out+="\\r"; break;
            case '\t': out+="\\t"; break;
            default:
                if((unsigned char)c < 0x20){
                    char buf[8]; snprintf(buf,sizeof(buf),"\\u%04x",c); out+=buf;
                } else out+=c;
        }
    }
    return out;
}

static std::string now_iso(){
    std::time_t t = std::time(nullptr);
    char buf[64];
    std::tm * gmt = std::gmtime(&t);
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", gmt);
    return std::string(buf);
}

static std::string role_from_name(const std::string & name){
    if(name.find("ffn_gate")!=std::string::npos) return "ffn_gate";
    if(name.find("ffn_up")!=std::string::npos) return "ffn_up";
    if(name.find("ffn_down")!=std::string::npos) return "ffn_down";
    if(name.find("attn_norm")!=std::string::npos) return "attn_norm";
    if(name.find("post_attention_norm")!=std::string::npos) return "post_attn_norm";
    if(name.find("ffn_norm")!=std::string::npos) return "ffn_norm";
    if(name.find("attn_q_norm")!=std::string::npos) return "attn_q_norm";
    if(name.find("attn_k_norm")!=std::string::npos) return "attn_k_norm";
    if(name.find("attn_q")!=std::string::npos) return "attn_q";
    if(name.find("attn_k")!=std::string::npos) return "attn_k";
    if(name.find("attn_v")!=std::string::npos) return "attn_v";
    if(name.find("attn_output")!=std::string::npos) return "attn_output";
    if(name.find("linear_attn_out")!=std::string::npos) return "linear_attn_out";
    if(name.find("ssm_conv")!=std::string::npos) return "ssm_conv";
    if(name.find("ssm_out")!=std::string::npos) return "ssm_out";
    if(name.find("ssm_in")!=std::string::npos) return "ssm_in";
    if(name.find("cache_k")!=std::string::npos) return "cache_k";
    if(name.find("cache_v")!=std::string::npos) return "cache_v";
    // gated delta net (linear attention) roles
    if(name.find("conv_output")!=std::string::npos) return "ssm_conv_out";
    if(name.find("q_conv")!=std::string::npos) return "q_conv";
    if(name.find("k_conv")!=std::string::npos) return "k_conv";
    if(name.find("v_conv")!=std::string::npos) return "v_conv";
    if(name.find("state_predelta")!=std::string::npos) return "ssm_state";
    if(name.find("beta_sigmoid")!=std::string::npos) return "beta_sigmoid";
    if(name.find("gate_sigmoid")!=std::string::npos) return "gate_sigmoid";
    if(name.find("a_softplus")!=std::string::npos) return "a_softplus";
    if(name.find("final_output")!=std::string::npos) return "final_output";
    // per-head projections and residual stream, as they appear in the rotation census
    if(name.find("Qcur_normed")!=std::string::npos) return "q_normed";
    if(name.find("Kcur_normed")!=std::string::npos) return "k_normed";
    if(name.find("Qcur_reshaped")!=std::string::npos) return "q_reshaped";
    if(name.find("Kcur_reshaped")!=std::string::npos) return "k_reshaped";
    if(name.find("Qcur")!=std::string::npos) return "attn_q";
    if(name.find("Kcur")!=std::string::npos) return "attn_k";
    if(name.find("attn_residual")!=std::string::npos) return "attn_residual";
    if(name.find("l_out")!=std::string::npos) return "l_out";
    if(name.find("input_embed")!=std::string::npos) return "input_embd";
    return "other";
}
static int layer_from_name(const std::string & name){
    size_t p = name.find("blk.");
    if(p!=std::string::npos){
        size_t start = p+4;
        size_t end = name.find(".", start);
        if(end!=std::string::npos){
            try { return std::stoi(name.substr(start, end-start)); } catch(...){}
        }
    }
    // trailing -N (e.g. ffn_gate-0)
    size_t dash = name.find_last_of("-");
    if(dash!=std::string::npos && dash+1 < name.size()){
        try { return std::stoi(name.substr(dash+1)); } catch(...){}
    }
    return -1;
}

static std::string type_name(ggml_type t){
    return std::string(ggml_type_name(t));
}

// ---------- rotation census (double-rotation guard) ----------
//
// A rotation-class op is one that changes the basis of its input: RoPE and the norm family
// (RMS/NORM/GROUP_NORM/L2_NORM). Applying two of them in a row, or two of the same kind to the
// same (layer, role), is how a fused op that gets re-scheduled into separately dispatched pieces
// silently rotates a tensor twice (the hipfire #897 "keep one binding alive for the entire layer"
// marker hazard, and the class the landed O11' norm fusion belongs to).
//
// This is a graph-build-time check only: it can see that a tensor is rotated twice, not that a
// kernel rotates its lanes by one element. It is the narrowest graph-visible form of the class.
static bool is_rotation_op(ggml_op op) {
    switch (op) {
        case GGML_OP_ROPE:
        case GGML_OP_RMS_NORM:
        case GGML_OP_NORM:
        case GGML_OP_GROUP_NORM:
        case GGML_OP_L2_NORM:
            return true;
        default:
            return false;
    }
}

// value-preserving ops that may sit between a rotation and its input after a fusion is re-scheduled
static bool is_identity_op(ggml_op op) {
    switch (op) {
        case GGML_OP_VIEW:
        case GGML_OP_RESHAPE:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_CONT:
            return true;
        default:
            return false;
    }
}

// the tensor a rotation node actually reads, looking through views/reshapes/copies
static ggml_tensor * rotation_origin(ggml_tensor * node) {
    ggml_tensor * t = node->src[0];
    for (int hop = 0; t != nullptr && is_identity_op(t->op) && hop < 8; hop++) {
        t = t->src[0];
    }
    return t;
}

struct rot_node {
    int          i;
    ggml_op      op;
    std::string  name;
    std::string  role;        // role of the rotated (input) tensor: "what was rotated"
    int          layer;       // layer of the rotation
    std::string  src0_name;   // name of that input tensor
    bool         src0_is_same_op;
};

static void collect_rotation_nodes(ggml_cgraph * gf, std::vector<rot_node> & out) {
    out.clear();
    const int n = ggml_graph_n_nodes(gf);
    for (int i = 0; i < n; i++) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        if (!is_rotation_op(node->op)) {
            continue;
        }
        // role/identity come from the immediate input (q_conv and k_conv are slices of one conv
        // output and must not collapse into one role), the chain check looks through views
        ggml_tensor * src0 = node->src[0];
        ggml_tensor * org  = rotation_origin(node);
        std::string role  = src0 ? role_from_name(src0->name) : "other";
        int layer = layer_from_name(node->name);
        if (role == "other") {
            role = role_from_name(node->name);
        }
        if (layer < 0 && src0) {
            layer = layer_from_name(src0->name);
        }
        out.push_back({
            i, node->op, node->name, role, layer,
            src0 ? src0->name : std::string(),
            org != nullptr && org->op == node->op,
        });
    }
}

// ---------- #936 fused-QKV split rule, static head geometry only ----------
//
// hipfire #936's head_mixer_admission reduced to what is decidable without any device:
//   - every weight the op reads must be planned          -> UnknownWeight
//   - all fused-QKV segments split on the same axis      -> UnshardableOp
//   - out_proj is a row split (axis 1), all else a head split -> UnsupportedPolicy
//   - heads % parts == 0 && kv_heads % parts == 0        -> LayoutMismatch
//   - parts == 1 means the output is local (Whole) and no reduction is emitted, so the
//     divisibility arms collapse - the same idea as the `n_rs == 1` direct fast path in build_rs
//
// Scope, deliberately narrow: this checks a rule against the model's head counts. It does not
// validate split *states* (segment layout / ne[] / nr[]), which the meta backend produces only
// with real devices; this host has one GPU and the internal all-reduce asserts n_backends == 2.

enum tp_verdict {
    TP_OK = 0,
    TP_ERR_LAYOUT_MISMATCH,
    TP_ERR_UNSHARDABLE_OP,
    TP_ERR_UNSUPPORTED_POLICY,
    TP_ERR_UNKNOWN_WEIGHT,
};

static const char * tp_verdict_name(tp_verdict v) {
    switch (v) {
        case TP_OK:                   return "ok";
        case TP_ERR_LAYOUT_MISMATCH:  return "LayoutMismatch";
        case TP_ERR_UNSHARDABLE_OP:   return "UnshardableOp";
        case TP_ERR_UNSUPPORTED_POLICY: return "UnsupportedPolicy";
        case TP_ERR_UNKNOWN_WEIGHT:   return "UnknownWeight";
    }
    return "?";
}

// one fused-QKV segment: axis of its shard, and whether the op has a slice of it at all
struct tp_seg {
    int  axis;     // 0 = head split
    bool planned;
};

static tp_verdict tp_qkv_admit(uint32_t parts, uint32_t heads, uint32_t kv_heads,
                               const std::vector<tp_seg> & segs, bool out_proj_row_axis1) {
    if (parts == 0 || !out_proj_row_axis1) {
        return TP_ERR_UNSUPPORTED_POLICY;
    }
    if (parts != 1 && (heads % parts != 0 || kv_heads % parts != 0)) {
        return TP_ERR_LAYOUT_MISMATCH;
    }
    int axis = -1;
    for (const tp_seg & s : segs) {
        if (!s.planned) {
            return TP_ERR_UNKNOWN_WEIGHT;
        }
        if (axis < 0) {
            axis = s.axis;
        } else if (s.axis != axis) {
            return TP_ERR_UNSHARDABLE_OP;
        }
    }
    return TP_OK;
}

static int run_tp_selftest() {
    struct c { const char * name; uint32_t parts, heads, kv; std::vector<tp_seg> segs; bool row; tp_verdict want; };
    const std::vector<c> cases = {
        { "valid_tp2",        2, 40,  8, {{0,true},{0,true},{0,true}}, true,  TP_OK },
        { "heads_not_divided",3, 40,  8, {{0,true},{0,true},{0,true}}, true,  TP_ERR_LAYOUT_MISMATCH },
        { "kv_not_divided",   2, 40,  3, {{0,true},{0,true},{0,true}}, true,  TP_ERR_LAYOUT_MISMATCH },
        { "mixed_axes",       2, 40,  8, {{0,true},{1,true},{0,true}}, true,  TP_ERR_UNSHARDABLE_OP },
        { "out_proj_col",     2, 40,  8, {{0,true},{0,true},{0,true}}, false, TP_ERR_UNSUPPORTED_POLICY },
        { "parts_zero",       0, 40,  8, {{0,true},{0,true},{0,true}}, true,  TP_ERR_UNSUPPORTED_POLICY },
        { "unplanned_weight", 2, 40,  8, {{0,true},{0,false},{0,true}}, true, TP_ERR_UNKNOWN_WEIGHT },
        { "parts_one_whole",  1,  7,  3, {{0,true},{0,true},{0,true}}, true,  TP_OK },
    };

    int n_bad = 0;
    for (const c & t : cases) {
        const tp_verdict got = tp_qkv_admit(t.parts, t.heads, t.kv, t.segs, t.row);
        const bool ok = got == t.want;
        n_bad += ok ? 0 : 1;
        printf("{\"type\":\"tp_selftest_case\",\"case\":\"%s\",\"want\":\"%s\",\"got\":\"%s\",\"ok\":%s}\n",
               t.name, tp_verdict_name(t.want), tp_verdict_name(got), ok ? "true" : "false");
    }
    printf("{\"type\":\"tp_selftest\",\"cases\":%zu,\"failed\":%d,\"ok\":%s}\n",
           cases.size(), n_bad, n_bad == 0 ? "true" : "false");
    return n_bad == 0 ? 0 : 6;
}

// ---------- q8_1 prequant lane-layout check (the O11' byte test) ----------
//
// The double-rotation census above is a graph-shape check: it can see a norm applied twice, but it
// cannot see a kernel that keeps the graph and the f32 output intact and only rotates the lanes of
// a side buffer. That is the O11' defect: rms_norm_f32's phase 2 writes the q8_1 blocks of the
// fused norm output (norm.cu) so that mmvq can skip its own quantise. Its contract is "the bytes
// are identical to quantize_q8_1 of the same f32 row"; a one-element rotation of the lane ->
// element mapping keeps every f32 value correct and only rotates those bytes. The failing byte
// test reported `byte 2 of 5760: a6 != 46`, i.e. the low byte of block 0's `s` (the f32 sum,
// which a rotation changes), in a buffer of 160 blocks * 36 bytes for n_embd = 5120.
//
// Narrowest check: compare that side buffer against the canonical quantizer, byte for byte,
// reporting the first mismatch. This is reference-free with respect to the *kernel under test*: the
// reference is a transcription of ggml/src/ggml-cuda/quantize.cu:quantize_q8_1, and the only
// ordering detail that matters is warp_reduce_sum's butterfly, which is reproduced below.
//
// Runnable without a GPU: --lane-selftest builds a synthetic row, quantises it with the reference
// and drives both negative controls (global one-lane rotation -> differs in `s` at byte 2;
// per-block qs rotation -> differs at byte 4). --lane-check <prefix> takes a dump of the real HIP
// side buffer (<prefix>.f32 = the fused norm output, <prefix>.q8_1 = the side buffer), so the
// verification itself needs no GPU; producing the dump does (GGML_HIP_PREQUANT_DUMP).

static const int QK8_1_SIZE  = 32;
static const int Q8_1_BYTES  = 36; // sizeof(block_q8_1) = 2*sizeof(ggml_half) + QK8_1

struct q8_1_cmp {
    size_t n_bytes     = 0;
    size_t n_mismatch  = 0; // differing bytes
    size_t n_bad       = 0; // differences a rotation explains: |dq| > 1, or any ds byte
    size_t first       = (size_t) -1;
    int    first_ref   = 0;
    int    first_got   = 0;
};

// host transcription of quantize_q8_1 for one contiguous f32 row
static void q8_1_ref_quantize(const float * x, int64_t ncols, std::vector<uint8_t> & out) {
    const int64_t nblocks = ncols / QK8_1_SIZE;
    out.assign((size_t) nblocks * Q8_1_BYTES, 0);
    for (int64_t ib = 0; ib < nblocks; ++ib) {
        float v[QK8_1_SIZE];
        for (int l = 0; l < QK8_1_SIZE; ++l) {
            v[l] = x[ib*QK8_1_SIZE + l];
        }
        float amax = 0.0f;
        for (int l = 0; l < QK8_1_SIZE; ++l) {
            amax = fmaxf(amax, fabsf(v[l]));
        }
        // warp_reduce_sum<QK8_1>: shfl_xor butterfly, offsets 16,8,4,2,1
        float t[QK8_1_SIZE];
        memcpy(t, v, sizeof(t));
        for (int off = QK8_1_SIZE/2; off > 0; off >>= 1) {
            float n[QK8_1_SIZE];
            for (int l = 0; l < QK8_1_SIZE; ++l) {
                n[l] = t[l] + t[l ^ off];
            }
            memcpy(t, n, sizeof(t));
        }
        const float d   = amax / 127.0f;
        const float sum = t[0];

        uint8_t * blk = out.data() + ib*Q8_1_BYTES;
        const ggml_fp16_t hd = ggml_fp32_to_fp16(d);
        const ggml_fp16_t hs = ggml_fp32_to_fp16(sum);
        memcpy(blk + 0, &hd, 2);
        memcpy(blk + 2, &hs, 2);
        for (int l = 0; l < QK8_1_SIZE; ++l) {
            blk[4 + l] = (uint8_t) (int8_t) (amax == 0.0f ? 0 : roundf(v[l] / d));
        }
    }
}

static q8_1_cmp q8_1_compare(const std::vector<uint8_t> & ref, const std::vector<uint8_t> & got) {
    q8_1_cmp r;
    r.n_bytes = std::min(ref.size(), got.size());
    const size_t nblocks = r.n_bytes / Q8_1_BYTES;
    const auto note = [&](size_t i) {
        if (r.first == (size_t) -1) {
            r.first     = i;
            r.first_ref = ref[i];
            r.first_got = got[i];
        }
    };
    const size_t n_blocks_end = nblocks*Q8_1_BYTES;
    if (ref.size() != got.size()) {
        r.n_mismatch += (ref.size() > got.size() ? ref.size() : got.size()) - n_blocks_end;
        r.n_bad      = r.n_mismatch; // a truncated/oversized buffer is never a rounding tie
        r.first      = n_blocks_end;
        r.first_ref  = n_blocks_end < ref.size() ? ref[n_blocks_end] : -1;
        r.first_got  = n_blocks_end < got.size() ? got[n_blocks_end] : -1;
        return r;
    }
    for (size_t ib = 0; ib < nblocks; ++ib) {
        // ds: d is amax/127 and s is the butterfly sum, both order independent; any difference is
        // a defect, never a rounding tie
        for (size_t b = 0; b < 4; ++b) {
            const size_t i = ib*Q8_1_BYTES + b;
            if (ref[i] != got[i]) {
                r.n_mismatch++;
                r.n_bad++;
                note(i);
            }
        }
        for (size_t l = 0; l < (size_t) QK8_1_SIZE; ++l) {
            const size_t i = ib*Q8_1_BYTES + 4 + l;
            if (ref[i] == got[i]) {
                continue;
            }
            r.n_mismatch++;
            // a +/-1 difference is the roundf tie window between host and device division; a
            // rotation moves whole lanes and is far outside it
            if (std::abs((int) (int8_t) ref[i] - (int) (int8_t) got[i]) > 1) {
                r.n_bad++;
            }
            note(i);
        }
    }
    return r;
}

static std::string q8_1_msg(const q8_1_cmp & c) {
    if (c.n_mismatch == 0) {
        return "identical";
    }
    char buf[128];
    snprintf(buf, sizeof(buf), "byte %zu of %zu: %02x != %02x",
             c.first, c.n_bytes, (unsigned) (uint8_t) c.first_ref, (unsigned) (uint8_t) c.first_got);
    return std::string(buf);
}

static std::vector<float> q8_1_test_row(int64_t ncols) {
    // spread the lanes so a rotation moves quantised values by tens, not by one: a smooth ramp
    // would stay inside the +/-1 tie window and make the detector blind
    std::vector<float> x(ncols);
    uint32_t s = 0x12345678u;
    for (int64_t i = 0; i < ncols; ++i) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        x[i] = (float) (s & 0xFFFFFFu) / 8388608.0f - 1.0f;
    }
    return x;
}

static int run_lane_selftest() {
    const int64_t ncols = 5120; // 27B n_embd: 160 blocks, 5760 bytes, the buffer from the report
    const std::vector<float> x = q8_1_test_row(ncols);

    std::vector<uint8_t> ref;
    q8_1_ref_quantize(x.data(), ncols, ref);

    // positive control: the reference against itself
    const q8_1_cmp c_pos = q8_1_compare(ref, ref);

    // negative control 1: the whole row rotated by one element (lane -> element mapping off by
    // one across block boundaries). d and s move with it, so this reproduces the reported `s` byte.
    std::vector<float> xr(ncols);
    for (int64_t i = 0; i < ncols; ++i) {
        xr[i] = x[(i + 1) % ncols];
    }
    std::vector<uint8_t> got_global;
    q8_1_ref_quantize(xr.data(), ncols, got_global);
    const q8_1_cmp c_global = q8_1_compare(ref, got_global);

    // negative control 2: qs lanes rotated by one inside each 32-element block (ds unchanged)
    std::vector<uint8_t> got_block = ref;
    for (size_t ib = 0; ib + Q8_1_BYTES <= got_block.size(); ib += Q8_1_BYTES) {
        for (int l = 0; l < QK8_1_SIZE; ++l) {
            got_block[ib + 4 + l] = ref[ib + 4 + ((l + 1) % QK8_1_SIZE)];
        }
    }
    const q8_1_cmp c_block = q8_1_compare(ref, got_block);

    const bool pos_ok       = c_pos.n_mismatch == 0;
    const bool global_caught = c_global.n_bad > 0;
    const bool block_caught  = c_block.n_bad > 0;

    printf("{\"type\":\"lane_selftest\",\"ncols\":%lld,\"n_bytes\":%zu,\"blocks\":%zu"
           ",\"positive_ok\":%s,\"global_rotation_caught\":%s,\"block_rotation_caught\":%s}\n",
           (long long) ncols, ref.size(), ref.size() / Q8_1_BYTES,
           pos_ok ? "true" : "false", global_caught ? "true" : "false",
           block_caught ? "true" : "false");
    if (c_global.n_mismatch) {
        printf("{\"type\":\"lane_selftest_case\",\"case\":\"global_rotation\",\"diff_bytes\":%zu"
               ",\"bad_bytes\":%zu,\"msg\":\"%s\"}\n",
               c_global.n_mismatch, c_global.n_bad, q8_1_msg(c_global).c_str());
    }
    if (c_block.n_mismatch) {
        printf("{\"type\":\"lane_selftest_case\",\"case\":\"block_rotation\",\"diff_bytes\":%zu"
               ",\"bad_bytes\":%zu,\"msg\":\"%s\"}\n",
               c_block.n_mismatch, c_block.n_bad, q8_1_msg(c_block).c_str());
    }
    return (pos_ok && global_caught && block_caught) ? 0 : 5;
}

static int run_lane_check(const std::string & prefix) {
    const auto read_all = [](const std::string & path, std::vector<uint8_t> & dst) {
        std::ifstream f(path, std::ios::binary);
        if (!f) {
            return false;
        }
        dst.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
        return true;
    };

    std::vector<uint8_t> f32_raw, got;
    if (!read_all(prefix + ".f32", f32_raw) || !read_all(prefix + ".q8_1", got)) {
        fprintf(stderr, "lane-check: cannot read %s.f32 / %s.q8_1\n", prefix.c_str(), prefix.c_str());
        return 2;
    }
    if (f32_raw.size() % sizeof(float) != 0) {
        fprintf(stderr, "lane-check: %s.f32 is not a float array\n", prefix.c_str());
        return 2;
    }
    const int64_t ncols = (int64_t) (f32_raw.size() / sizeof(float));
    if (got.size() != (size_t) (ncols / QK8_1_SIZE) * Q8_1_BYTES) {
        fprintf(stderr, "lane-check: %s.q8_1 is %zu bytes, expected %zu for %lld f32 values\n",
                prefix.c_str(), got.size(), (size_t) (ncols / QK8_1_SIZE) * Q8_1_BYTES, (long long) ncols);
        return 2;
    }

    std::vector<uint8_t> ref;
    q8_1_ref_quantize(reinterpret_cast<const float *>(f32_raw.data()), ncols, ref);
    const q8_1_cmp c = q8_1_compare(ref, got);

    printf("{\"type\":\"lane_check\",\"prefix\":\"%s\",\"ncols\":%lld,\"n_bytes\":%zu"
           ",\"diff_bytes\":%zu,\"bad_bytes\":%zu,\"first\":%zu"
           ",\"first_ref\":%d,\"first_got\":%d,\"msg\":\"%s\",\"ok\":%s}\n",
           json_escape(prefix).c_str(), (long long) ncols, c.n_bytes, c.n_mismatch, c.n_bad,
           c.first, c.first_ref, c.first_got, q8_1_msg(c).c_str(), c.n_bad == 0 ? "true" : "false");

    if (c.n_bad != 0) {
        fprintf(stderr, "lane-check: FAIL - q8_1 side buffer is not the canonical quantisation (%s)\n",
                q8_1_msg(c).c_str());
        return 3;
    }
    return 0;
}

int main(int argc, char ** argv) {
    // lane-layout checks: no model, no params parsing (and no GPU for the self-test)
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--tp-selftest") {
            return run_tp_selftest();
        }
        if (a == "--lane-selftest") {
            return run_lane_selftest();
        }
        if (a == "--lane-check") {
            if (i + 1 >= argc) {
                fprintf(stderr, "usage: %s --lane-check <prefix>   (expects <prefix>.f32 and <prefix>.q8_1)\n", argv[0]);
                return 2;
            }
            return run_lane_check(argv[i + 1]);
        }
    }

    // Negative control: build a tiny graph with a double rotation by hand and check that the same
    // detector used on a real census fires. A guard that never fires is not a guard.
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) != "--selftest") {
            continue;
        }
        ggml_init_params ip = { 1024*1024, nullptr, /*.no_alloc =*/ true };
        ggml_context * c = ggml_init(ip);
        ggml_tensor * x = ggml_new_tensor_1d(c, GGML_TYPE_F32, 8);
        ggml_set_name(x, "x");

        // chain: norm(norm(x))
        ggml_tensor * a = ggml_rms_norm(c, x, 1e-6f);
        ggml_set_name(a, "norm_a");
        ggml_tensor * b = ggml_rms_norm(c, a, 1e-6f);
        ggml_set_name(b, "norm_b");

        // duplicate: two siblings both rotating the same input
        ggml_tensor * d1 = ggml_rms_norm(c, x, 1e-6f);
        ggml_set_name(d1, "norm_d1");
        ggml_tensor * d2 = ggml_rms_norm(c, x, 1e-6f);
        ggml_set_name(d2, "norm_d2");

        ggml_cgraph * g = ggml_new_graph(c);
        ggml_build_forward_expand(g, b);
        ggml_build_forward_expand(g, d1);
        ggml_build_forward_expand(g, d2);

        std::vector<rot_node> r;
        collect_rotation_nodes(g, r);
        bool chain = false;
        int  dup   = 0;
        std::map<std::string, int> per_input;
        for (const auto & rn : r) {
            chain = chain || rn.src0_is_same_op;
            dup = std::max(dup, ++per_input[rn.src0_name]);
        }
        ggml_free(c);
        printf("{\"type\":\"selftest\",\"rotation_nodes\":%zu,\"chain_detected\":%s,"
               "\"duplicate_detected\":%s}\n", r.size(), chain ? "true" : "false",
               dup > 1 ? "true" : "false");
        return (chain && dup > 1) ? 0 : 5;
    }

    // custom pre-parse
    std::vector<std::string> args_vec;
    std::string mode = "pp";
    bool jsonl = true;
    bool verify = false;
    std::string out_path = "";
    for(int i=1;i<argc;i++){
        std::string a = argv[i];
        if(a=="--mode"){ if(i+1<argc){ mode=argv[++i]; } continue; }
        if(a=="--raw"||a=="--jsonl"){ jsonl=true; continue; }
        if(a=="--verify"){ verify=true; continue; }
        if(a=="--out"){ if(i+1<argc){ out_path=argv[++i]; } continue; }
        args_vec.push_back(a);
    }
    // rebuild argc/argv
    std::vector<char*> nargv;
    nargv.push_back(argv[0]);
    for(auto & s: args_vec){ nargv.push_back(const_cast<char*>(s.c_str())); }
    int nargc = nargv.size();
    char ** argv2 = nargv.data();

    common_params params;
    params.out_file = "tests.txt";
    common_init();
    if (!common_params_parse(nargc, argv2, params, LLAMA_EXAMPLE_EXPORT_GRAPH_OPS)) {
        return 1;
    }
    ggml_backend_dev_t cpu_device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    params.devices = { cpu_device, nullptr };
    params.fit_params = false;
    params.n_gpu_layers = 0;
    params.warmup = false;

    llama_context * ctx;
    common_init_result_ptr init_result;
    llama_context_ptr ctx2;
    llama_model_ptr model;

    if (params.model.hf_repo.empty()) {
        init_result = common_init_from_params(params);
        ctx = init_result->context();
        if (!ctx) { LOG_ERR("failed to initialize params\n"); return 1; }
    } else {
#ifdef LLAMA_HF_FETCH
        auto [hf_repo, hf_quant] = common_download_split_repo_tag(params.model.hf_repo);
        if (hf_quant.empty() || hf_quant == "latest") hf_quant = "Q4_K_M";
        gguf_context_ptr gguf_ctx = gguf_fetch_gguf_ctx(hf_repo, hf_quant);
        if (!gguf_ctx) { LOG_ERR("failed to fetch GGUF metadata\n"); return 1; }
        llama_model_params model_params = llama_model_default_params();
        model_params.devices = params.devices.data();
        model_params.no_alloc = true;
        model.reset(llama_model_init_from_user(gguf_ctx.get(), set_tensor_data, nullptr, model_params));
        if (!model) { LOG_ERR("failed to create model\n"); return 1; }
        llama_context_params ctx_params = llama_context_default_params();
        ctx2.reset(llama_init_from_model(model.get(), ctx_params));
        ctx = ctx2.get();
        if (!ctx) { LOG_ERR("failed to create ctx\n"); return 1; }
#else
        LOG_ERR("no hf fetch\n"); return 1;
#endif
    }

    const uint32_t n_seqs  = llama_n_seq_max(ctx);
    const uint32_t n_ubatch = llama_n_ubatch(ctx);
    const uint32_t n_ctx = llama_n_ctx(ctx);

    uint32_t n_tokens;
    if (mode == "tg") {
        n_tokens = n_seqs;
    } else {
        n_tokens = std::min(n_ctx, n_ubatch);
    }
    // llama_graph_reserve() invalidates the previous graph, so every caller has to rebuild
    auto reserve_graph = [&]() -> ggml_cgraph * {
        if (mode == "tg") {
            return llama_graph_reserve(ctx, n_seqs, n_seqs, n_seqs);
        }
        return llama_graph_reserve(ctx, n_tokens, n_seqs, n_tokens);
    };
    ggml_cgraph * gf = reserve_graph();
    if (!gf) { LOG_ERR("failed to reserve graph\n"); return 1; }

    std::vector<rot_node> rots;
    collect_rotation_nodes(gf, rots);

    int n_nodes = ggml_graph_n_nodes(gf);
    int n_skipped = 0;
    for (int i = 0; i < n_nodes; i++) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        if (node->op == GGML_OP_NONE || node->op == GGML_OP_VIEW || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_TRANSPOSE) {
            n_skipped++;
        }
    }
    int n_real = n_nodes - n_skipped;

    // model meta
    const llama_model * model_ptr = llama_get_model(ctx);
    std::string model_path = params.model.path;
    std::string arch = "unknown";
    {
        gguf_init_params gp = { true, nullptr };
        gguf_context * gctx = gguf_init_from_file(model_path.c_str(), gp);
        if (gctx) {
            int64_t aid = gguf_find_key(gctx, "general.architecture");
            if (aid >= 0) {
                const char * s = gguf_get_val_str(gctx, aid);
                if (s) arch = s;
            }
            gguf_free(gctx);
        }
    }
    int n_layer = (int)llama_model_n_layer(model_ptr) + (int)llama_model_n_layer_nextn(model_ptr);

    // model sha256
    std::string model_sha = sha256_file(model_path);
    std::string binary_sha = sha256_file("/home/stormrage/llama.cpp/build-hip-gfx1030/bin/libggml-hip.so.0");

    // output stream
    std::ostream * out = &std::cout;
    std::ofstream fout;
    if (!out_path.empty()) {
        fout.open(out_path);
        out = &fout;
    }

    // meta
    *out << "{\"type\":\"meta\",\"schema\":1";
    *out << ",\"model\":\"" << json_escape(model_path) << "\"";
    *out << ",\"model_sha256\":\"" << model_sha << "\"";
    *out << ",\"binary_sha256\":\"" << binary_sha << "\"";
    *out << ",\"arch\":\"" << arch << "\"";
    *out << ",\"n_layer\":" << n_layer;
    *out << ",\"n_ctx\":" << n_ctx;
    *out << ",\"n_ubatch\":" << n_ubatch;
    *out << ",\"n_seqs\":" << n_seqs;
    *out << ",\"mode\":\"" << mode << "\"";
    *out << ",\"n_nodes\":" << n_nodes;
    *out << ",\"n_skipped_views\":" << n_skipped;
    *out << ",\"n_real\":" << n_real;
    *out << ",\"created\":\"" << now_iso() << "\"";
    *out << "}\n";

    // static #936 split-rule census over this model's head geometry: a fused-QKV plan with one
    // planned Q/K/V segment, all head-sharded on axis 0, out_proj row-sharded on axis 1.
    // No ranks, no devices, no numerics - see the scope note next to tp_qkv_admit().
    {
        const int32_t n_head    = llama_model_n_head(model_ptr);
        const int32_t n_head_kv = llama_model_n_head_kv(model_ptr);
        for (uint32_t parts = 1; parts <= 8; parts *= 2) {
            const tp_verdict v = (n_head <= 0 || n_head_kv <= 0) ? TP_ERR_UNSUPPORTED_POLICY :
                tp_qkv_admit(parts, (uint32_t) n_head, (uint32_t) n_head_kv,
                             std::vector<tp_seg>{{0,true},{0,true},{0,true}}, true);
            *out << "{\"type\":\"tp_geometry\",\"parts\":" << parts
                 << ",\"n_head\":" << n_head << ",\"n_head_kv\":" << n_head_kv
                 << ",\"verdict\":\"" << tp_verdict_name(v) << "\""
                 << ",\"admitted\":" << (v == TP_OK ? "true" : "false")
                 << ",\"scope\":\"head-counts-only\"}\n";
        }
    }

    // map tensor ptr -> node index
    std::map<ggml_tensor*, int> idxmap;
    for (int i = 0; i < n_nodes; i++) {
        idxmap[ggml_graph_node(gf, i)] = i;
    }

    for (int i = 0; i < n_nodes; i++) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        // src list
        std::vector<ggml_tensor*> srcs;
        for (size_t s = 0; s < GGML_MAX_SRC; s++) {
            if (node->src[s] == nullptr) break;
            srcs.push_back(node->src[s]);
        }

        // weight detection
        ggml_tensor * w = nullptr;
        for (auto * t : srcs) {
            if (t->flags & GGML_TENSOR_FLAG_PARAM) { w = t; break; }
        }
        if (!w) {
            for (auto * t : srcs) {
                std::string nm = t->name;
                if (nm.size() >= 7 && nm.substr(nm.size()-7) == ".weight") { w = t; break; }
            }
        }

        std::string name = node->name;

        *out << "{\"type\":\"node\"";
        *out << ",\"i\":" << i;
        *out << ",\"op\":" << (int)node->op;
        *out << ",\"op_name\":\"" << json_escape(ggml_op_name(node->op)) << "\"";
        *out << ",\"name\":\"" << json_escape(name) << "\"";
        *out << ",\"type\":" << (int)node->type;
        *out << ",\"type_name\":\"" << json_escape(type_name(node->type)) << "\"";
        *out << ",\"ne\":[" << node->ne[0] << "," << node->ne[1] << "," << node->ne[2] << "," << node->ne[3] << "]";
        *out << ",\"nb\":[" << node->nb[0] << "," << node->nb[1] << "," << node->nb[2] << "," << node->nb[3] << "]";
        *out << ",\"nbytes\":" << ggml_nbytes(node);
        *out << ",\"flags\":{\"contiguous\":" << (ggml_is_contiguous(node)?"true":"false")
             << ",\"transposed\":" << (ggml_is_transposed(node)?"true":"false")
             << ",\"view\":" << (node->view_src != nullptr ? "true" : "false") << "}";
        // src
        *out << ",\"src\":[";
        for (size_t s = 0; s < srcs.size(); s++) {
            if (s) *out << ",";
            ggml_tensor * t = srcs[s];
            int idx = -1;
            auto it = idxmap.find(t);
            if (it != idxmap.end()) idx = it->second;
            *out << "{\"idx\":" << idx;
            *out << ",\"name\":\"" << json_escape(t->name) << "\"";
            *out << ",\"op\":" << (int)t->op;
            *out << ",\"type\":" << (int)t->type;
            *out << ",\"type_name\":\"" << json_escape(type_name(t->type)) << "\"";
            *out << ",\"ne\":[" << t->ne[0] << "," << t->ne[1] << "," << t->ne[2] << "," << t->ne[3] << "]";
            *out << ",\"nbytes\":" << ggml_nbytes(t);
            *out << "}";
        }
        *out << "]";
        if (w) {
            *out << ",\"weight\":{\"name\":\"" << json_escape(w->name) << "\",\"type\":" << (int)w->type
                 << ",\"type_name\":\"" << json_escape(type_name(w->type)) << "\"}";
        }
        std::string r = role_from_name(name);
        int l = layer_from_name(name);
        if (l < 0) {
            for (auto * t : srcs) {
                l = layer_from_name(t->name);
                if (l >= 0) break;
            }
        }
        if (l < 0 && w) l = layer_from_name(w->name);
        *out << ",\"role\":\"" << r << "\"";
        *out << ",\"layer\":" << l;
        *out << "}\n";
    }

    // ---------- rotation census: per (layer, role, op) counts, then the double-rotation checks ----
    //
    // Deterministic output (sorted by layer, role, op) so two builds can be diffed directly.
    std::map<std::tuple<int, std::string, std::string>, std::vector<const rot_node *>> groups;
    for (const auto & r : rots) {
        groups[std::make_tuple(r.layer, r.role, std::string(ggml_op_name(r.op)))].push_back(&r);
    }

    int n_violations = 0;
    for (const auto & kv : groups) {
        const int         layer = std::get<0>(kv.first);
        const std::string & role = std::get<1>(kv.first);
        const std::string & op_n = std::get<2>(kv.first);
        *out << "{\"type\":\"rotation\",\"layer\":" << layer
             << ",\"role\":\"" << json_escape(role) << "\""
             << ",\"op_name\":\"" << json_escape(op_n) << "\""
             << ",\"count\":" << kv.second.size() << "}\n";
    }

    // (a) the same input tensor rotated twice by the same op. A genuine re-schedule that duplicates
    //     an op instead of chaining it lands here; distinct slices of one fused tensor (q_conv vs
    //     k_conv) have distinct names and stay separate.
    std::map<std::pair<std::string, std::string>, std::vector<const rot_node *>> inputs;
    for (const auto & r : rots) {
        if (r.src0_name.empty()) {
            continue;
        }
        inputs[std::make_pair(std::string(ggml_op_name(r.op)), r.src0_name)].push_back(&r);
    }
    for (const auto & kv : inputs) {
        if (kv.second.size() < 2) {
            continue;
        }
        n_violations++;
        *out << "{\"type\":\"rotation_violation\",\"kind\":\"duplicate\""
             << ",\"op_name\":\"" << json_escape(kv.first.first) << "\""
             << ",\"src_name\":\"" << json_escape(kv.first.second) << "\""
             << ",\"count\":" << kv.second.size() << ",\"names\":[";
        for (size_t k = 0; k < kv.second.size(); k++) {
            if (k) *out << ",";
            *out << "\"" << json_escape(kv.second[k]->name) << "\"";
        }
        *out << "]}\n";
    }

    // (b) a rotation-class op applied directly on top of the same rotation-class op: norm(norm(x))
    //     or rope(rope(x)). This is the actual double-application signature; it needs no role label.
    for (const auto & r : rots) {
        if (!r.src0_is_same_op) {
            continue;
        }
        n_violations++;
        *out << "{\"type\":\"rotation_violation\",\"kind\":\"chain\",\"layer\":" << r.layer
             << ",\"role\":\"" << json_escape(r.role) << "\""
             << ",\"op_name\":\"" << json_escape(ggml_op_name(r.op)) << "\""
             << ",\"name\":\"" << json_escape(r.name) << "\""
             << ",\"src_name\":\"" << json_escape(r.src0_name) << "\"}\n";
    }

    bool verify_ok = true;
    if (verify) {
        // Build the same graph a second time and compare the rotation census. The graph builder must
        // be deterministic in which tensors it rotates; a mismatch means the census (or the builder)
        // is load-order dependent and cannot be used as a regression guard.
        ggml_cgraph * gf2 = reserve_graph();
        if (!gf2) { LOG_ERR("failed to reserve graph (verify)\n"); return 1; }
        std::vector<rot_node> rots2;
        collect_rotation_nodes(gf2, rots2);
        std::map<std::tuple<int, std::string, std::string>, size_t> groups2;
        for (const auto & r : rots2) {
            groups2[std::make_tuple(r.layer, r.role, std::string(ggml_op_name(r.op)))]++;
        }
        if (groups2.size() != groups.size()) {
            verify_ok = false;
        } else {
            for (const auto & kv : groups) {
                auto it = groups2.find(kv.first);
                if (it == groups2.end() || it->second != kv.second.size()) {
                    verify_ok = false;
                    break;
                }
            }
        }
        *out << "{\"type\":\"rotation_verify\",\"builds\":2,\"n_rotation_nodes\":["
             << rots.size() << "," << rots2.size() << "],\"ok\":"
             << (verify_ok ? "true" : "false") << "}\n";
    }

    *out << "{\"type\":\"rotation_summary\",\"schema\":1"
         << ",\"n_rotation_nodes\":" << rots.size()
         << ",\"violations\":" << n_violations
         << ",\"ok\":" << ((n_violations == 0 && verify_ok) ? "true" : "false") << "}\n";

    if (fout.is_open()) fout.close();
    if (n_violations != 0) {
        LOG_ERR("rotation census: %d double-rotation violation(s)\n", n_violations);
        return 3;
    }
    if (!verify_ok) {
        LOG_ERR("rotation census: graph build is not deterministic across builds\n");
        return 4;
    }
    return 0;
}
