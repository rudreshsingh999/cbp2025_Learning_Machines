#ifndef _TAGE_RL_PREDICTOR_H_
#define _TAGE_RL_PREDICTOR_H_

#include <iostream>
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <array>
#include <cmath>
#include <cassert>
#include <cstdlib>
#include <algorithm>

#define GHIST_LENGTH 32
#define LHIST_LENGTH 8


static inline float clip(float x, float lo, float hi) {
    return std::min(std::max(x, lo), hi);
}

static inline float sigmoid(float x) {
    return 1.f / (1.f + std::exp(-x));
}

static inline float randf() {
    return static_cast<float>(std::rand()) / RAND_MAX;
}

struct RLPolicyNet {
    std::vector<float> weights;
    float bias = 0.f;
    float lr = 0.02f;
    const float w_clip = 0.5f;
    const float lr_min = 0.002f;
    const float lr_decay = 0.9999f;

    RLPolicyNet(size_t input_size) {
        weights.assign(input_size, 0.02f);
    }

    float forward(const std::vector<int>& input) const {
        assert(input.size() == weights.size());
        float x = bias;
        for (size_t i = 0; i < weights.size(); i++)
            x += weights[i] * input[i];
        return sigmoid(x);
    }

    void update(const std::vector<int>& input, float logp, float reward) {
        float baseline = 0.f;
        const float beta = 0.99f;
        // float grad = -reward;  // ∇(-logp * reward)
        float advantage = reward - baseline;
        baseline = beta * baseline + (1 - beta) * reward;  // moving average baseline
        float grad = -advantage;
        for (size_t i = 0; i < weights.size(); i++)
            weights[i] = clip(weights[i] - lr * grad * input[i], -w_clip, w_clip);
        bias = clip(bias - lr * grad, -w_clip, w_clip);
        lr = std::max(lr * lr_decay, lr_min);
    }
};

struct RLHist {
    uint64_t ghist;
    std::vector<int> input_bits;
    float logp;
    bool action;
};

struct TAGEQualityTable {
    std::array<uint8_t, 4096> q{};
    inline bool weak(uint16_t idx) const { return q[idx] < 8; }
    void update(uint16_t idx, bool tage_correct) {
        if (tage_correct) {
            if (q[idx] < 15) q[idx]++;
        } else {
            if (q[idx] > 0) q[idx]--;
        }
    }
};

class TAGE_RL_Predictor {
    RLPolicyNet policy;
    TAGEQualityTable tage_q;
    std::unordered_map<uint64_t, RLHist> pred_time_histories;
    uint64_t ghist = 0;

public:
    TAGE_RL_Predictor() : policy(GHIST_LENGTH + LHIST_LENGTH) {
        std::srand(1);
    }

    void setup() {}
    void terminate() {}

    uint64_t get_key(uint64_t seq_no, uint8_t piece) const {
        return (seq_no << 4) | (piece & 0x0F);
    }

    std::vector<int> encode_bits(uint64_t g, uint64_t l) const {
        std::vector<int> bits;
        for (int i = 0; i < GHIST_LENGTH; i++)
            bits.push_back((g >> i) & 1 ? 1 : -1);
        for (int i = 0; i < LHIST_LENGTH; i++)
            bits.push_back((l >> i) & 1 ? 1 : -1);
        return bits;
    }

    bool predict(uint64_t seq_no, uint8_t piece, uint64_t PC, bool tage_pred) {
        const uint64_t key = get_key(seq_no, piece);
        uint64_t lhist = PC & ((1ULL << LHIST_LENGTH) - 1);
        std::vector<int> input = encode_bits(ghist, lhist);

        float prob = policy.forward(input);
        bool action = randf() < prob;
        
        // RLHist hist{ghist, input, std::log(prob + 1e-5f), action};
        float safe_prob = std::max(1e-5f, std::min(0.99999f, prob));
        float logp = std::log(safe_prob);
        RLHist hist{ghist, input, logp, action};

        pred_time_histories[key] = hist;
        uint16_t idx = (uint16_t)(((PC >> 2) ^ PC) & 0xFFF);
        bool use_rl = tage_q.weak(idx) && (prob > 0.9f || prob < 0.1f);

        return use_rl ? action : tage_pred;
    }

    void history_update(uint64_t seq_no, uint8_t piece, uint64_t PC, bool taken, uint64_t nextPC) {
        ghist = ((ghist << 1) | (taken ? 1 : 0)) & ((1ULL << GHIST_LENGTH) - 1);
    }
    void update(uint64_t seq_no, uint8_t piece, uint64_t PC, bool resolveDir, bool pred_taken, uint64_t nextPC) {
        const uint64_t key = get_key(seq_no, piece);

        auto it = pred_time_histories.find(key);
        if (it == pred_time_histories.end())
            return;

        RLHist hist = it->second; 
        pred_time_histories.erase(it);

        if (hist.input_bits.size() != policy.weights.size()) {
            std::cerr << "[WARN] input_bits.size() != weights.size() → " 
                    << hist.input_bits.size() << " vs " << policy.weights.size() << std::endl;
            return;
        }

        if (!std::isfinite(hist.logp)) {
            std::cerr << "[WARN] logp is not finite!" << std::endl;
            return;
        }

        uint16_t idx = (uint16_t)(((PC >> 2) ^ PC) & 0xFFF);
        tage_q.update(idx, pred_taken == resolveDir);

        float reward = (hist.action == resolveDir) ? 1.0f : -1.0f;
        policy.update(hist.input_bits, hist.logp, reward);
    }


    void update(uint64_t PC, bool resolveDir, bool pred_taken, uint64_t nextPC, const RLHist& hist_unused) {
        // unused
    }
};

static TAGE_RL_Predictor cond_predictor_impl;

#endif
