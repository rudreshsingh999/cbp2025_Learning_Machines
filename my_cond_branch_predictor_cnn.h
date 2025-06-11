#ifndef _PREDICTOR_H_
#define _PREDICTOR_H_

#include <unordered_map>
#include <vector>
#include <cstdint>
#include <cassert>
#include <cmath>
#include <algorithm>

#define GHIST_LENGTH 32
#define LHIST_LENGTH 8
#define PERCEPTRON_THRESHOLD 20
#define MAX_WEIGHT 127
#define MIN_WEIGHT -128

static inline float clip(float x,float lo,float hi){return std::min(std::max(x,lo),hi);}  
static inline float lrelu(float v){return (v>0.f)?v:0.1f*v;}  

struct Slice {
    int hist_len;
    std::vector<float> conv;
    float bias = 0.0f;
};

class CNNPredictor {
    std::vector<Slice>  slices;
    std::vector<float>  fc_w;
    float               fc_bias = 0.f;
    std::array<float,4096> pc_bias;
    float lr = 0.02f;
    const float lr_min = 0.002f;
    const float lr_decay = 0.99995f;
    const float w_clip = 0.5f;
    const float base_thr = 1.5f;

    std::vector<int8_t>  last_bits;
    std::vector<float>   last_slice;
    uint16_t             last_pc_idx=0;
    std::unordered_map<uint64_t,uint16_t> local_hist;

public:
    CNNPredictor() {
        for(int L:{8,16,32,64})
            slices.push_back({L,{0.04f,0.04f,0.04f},0.0f});
        fc_w.assign(slices.size(),0.04f);
        pc_bias.fill(0.f);
        std::srand(1);
    }

    float compute(uint64_t ghist, uint64_t PC, bool& pred) {
        last_bits.clear();
        for(auto &s:slices){
            for(int i=0;i<s.hist_len;i++)
                last_bits.push_back(((ghist>>i)&1)?1:-1);
        }
        uint16_t lh = local_hist[PC];
        for(int i=0;i<LHIST_LENGTH;i++)
            last_bits.push_back(((lh>>i)&1)?1:-1);

        last_slice.clear();
        int pos=0;
        for(auto &s:slices){
            float pool=0.f;
            for(int i=0;i+s.conv.size()<=s.hist_len;i++){
                float v=s.bias;
                for(int k=0;k<3;k++) v += s.conv[k]*last_bits[pos+i+k];
                pool += lrelu(v);
            }
            last_slice.push_back(pool);
            pos += s.hist_len;
        }

        last_pc_idx = (uint16_t)(((PC>>2) ^ PC) & 0xFFF);
        float logit = fc_bias + pc_bias[last_pc_idx];
        for(size_t i=0;i<fc_w.size();i++) logit += fc_w[i]*last_slice[i];
        pred = logit>=0.f;
        return logit;
    }

    void update(uint64_t ghist, uint64_t PC, bool taken, float logit) {
        int target = taken?1:-1;
        float conf_thr = base_thr*std::sqrt((float)last_slice.size());
        bool update_needed = (taken!=(logit>=0.f)) || std::fabs(logit)<=conf_thr;
        local_hist[PC] = ((local_hist[PC]<<1)|(taken?1:0)) & ((1u<<LHIST_LENGTH)-1);
        if(!update_needed){ return; }

        pc_bias[last_pc_idx] = clip(pc_bias[last_pc_idx] + lr*target, -w_clip, w_clip);
        for(size_t i=0;i<fc_w.size();i++)
            fc_w[i] = clip(fc_w[i] + lr*target*last_slice[i],-w_clip,w_clip);
        fc_bias = clip(fc_bias + lr*target,-w_clip,w_clip);

        int pos=0;
        for(size_t sidx=0;sidx<slices.size();sidx++){
            auto &s = slices[sidx];
            for(int i=0;i+s.conv.size()<=s.hist_len;i++){
                float pre = s.bias;
                for(int k=0;k<3;k++) pre += s.conv[k]*last_bits[pos+i+k];
                float grad_relu = (pre>0.f)?1.f:0.1f;
                float g = lr*target*fc_w[sidx]*grad_relu;
                for(int k=0;k<3;k++){
                    int bit = last_bits[pos+i+k];
                    s.conv[k] = clip(s.conv[k] + g*bit,-w_clip,w_clip);
                }
                s.bias = clip(s.bias + g,-w_clip,w_clip);
            }
            pos += s.hist_len;
        }
        lr = std::max(lr*lr_decay, lr_min);
    }
};

struct SampleHist {
    uint64_t ghist;
    bool tage_pred;
    float cnn_logit;
    SampleHist() : ghist(0), tage_pred(false), cnn_logit(0.f) {}
};

struct TAGEQualityTable {
    std::array<uint8_t, 4096> q{};
    inline bool weak(uint16_t idx) const { return q[idx] < 8; }
    void update(uint16_t idx, bool tage_correct) {
        if(tage_correct) {
            if(q[idx]<15) q[idx]++;
        } else {
            if(q[idx]>0)  q[idx]--;
        }
    }
};

class SampleCondPredictor {
    SampleHist active_hist;
    std::unordered_map<uint64_t, SampleHist> pred_time_histories;
    CNNPredictor predictor;
    TAGEQualityTable tage_q;

public:
    SampleCondPredictor() {}

    void setup() {}
    void terminate() {}

    uint64_t get_unique_inst_id(uint64_t seq_no, uint8_t piece) const {
        assert(piece < 16);
        return (seq_no << 4) | (piece & 0x000F);
    }

    bool predict(uint64_t seq_no, uint8_t piece, uint64_t PC, const bool tage_pred) {
        active_hist.tage_pred = tage_pred;
        const uint64_t key = get_unique_inst_id(seq_no, piece);
        pred_time_histories[key] = active_hist;

        bool cnn_pred = false;
        float logit = predictor.compute(active_hist.ghist, PC, cnn_pred);
        active_hist.cnn_logit = logit;

        uint16_t idx = (uint16_t)(((PC>>2)^PC)&0xFFF);
        bool use_cnn = tage_q.weak(idx) && std::fabs(logit) > 1.2f;

        return use_cnn ? cnn_pred : tage_pred;
    }

    void history_update(uint64_t seq_no, uint8_t piece, uint64_t PC, bool taken, uint64_t nextPC) {
        active_hist.ghist = (active_hist.ghist << 1 | (taken ? 1 : 0)) & ((1ULL << GHIST_LENGTH) - 1);
    }

    void update(uint64_t seq_no, uint8_t piece, uint64_t PC, bool resolveDir, bool predDir, uint64_t nextPC) {
        const auto key = get_unique_inst_id(seq_no, piece);
        const auto& hist = pred_time_histories.at(key);
        pred_time_histories.erase(key);

        uint16_t idx = (uint16_t)(((PC>>2)^PC)&0xFFF);
        tage_q.update(idx, hist.tage_pred == resolveDir);

        predictor.update(hist.ghist, PC, resolveDir, hist.cnn_logit);
    }

    void update(uint64_t PC, bool resolveDir, bool pred_taken, uint64_t nextPC, const SampleHist& hist_to_use) {
        // unused
    }
};

static SampleCondPredictor cond_predictor_impl;

#endif
