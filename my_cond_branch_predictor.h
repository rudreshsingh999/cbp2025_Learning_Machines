#ifndef _PREDICTOR_H_
#define _PREDICTOR_H_

#include <unordered_map>
#include <vector>
#include <cstdint>
#include <cassert>
#include <cmath>

#define GHIST_LENGTH 32
#define LHIST_LENGTH 8
#define PERCEPTRON_THRESHOLD 20
#define MAX_WEIGHT 127
#define MIN_WEIGHT -128

// Struct for Sample History
struct SampleHist {
    uint64_t ghist;

    SampleHist() : ghist(0) {}
};

// Perceptron Predictor Class
class PerceptronPredictor {
    using Weights = std::vector<int8_t>;

    std::unordered_map<uint64_t, Weights> table;
    std::unordered_map<uint64_t, uint16_t> local_hist; // Per-PC local history

public:
    PerceptronPredictor() {}

    
    int compute(uint64_t ghist, uint64_t PC, bool& pred) {
        const uint64_t index = (PC >> 1) ^ (PC >> 13) ^ (PC >> 5) % 4096;
        auto& weights = table[index];

        if (weights.empty())
            weights.resize(GHIST_LENGTH + LHIST_LENGTH + 1, 0); // +1 for bias

        int sum = weights[0]; // bias term

        // Global history contribution
        for (int i = 0; i < GHIST_LENGTH; ++i) {
            int bit = (ghist >> i) & 1;
            sum += weights[i + 1] * (bit ? 1 : -1);
        }

        // Local history contribution
        uint16_t lhist = local_hist[PC];
        for (int i = 0; i < LHIST_LENGTH; ++i) {
            int bit = (lhist >> i) & 1;
            sum += weights[GHIST_LENGTH + 1 + i] * (bit ? 1 : -1);
        }

        pred = (sum >= 0);
        return sum;
    }

    void update(uint64_t ghist, uint64_t PC, bool taken, int sum) {
        const uint64_t index = (PC >> 1) ^ (PC >> 13) ^ (PC >> 5) % 4096;
        auto& weights = table[index];

        if (weights.empty())
            weights.resize(GHIST_LENGTH + LHIST_LENGTH + 1, 0);

        int target = taken ? 1 : -1;

        if ((taken != (sum >= 0)) || std::abs(sum) <= PERCEPTRON_THRESHOLD) {
            // Bias
            int tmp = weights[0] + target;
            if (tmp > MAX_WEIGHT) tmp = MAX_WEIGHT;
            if (tmp < MIN_WEIGHT) tmp = MIN_WEIGHT;
            weights[0] = tmp;

            // Global history weights
            for (int i = 0; i < GHIST_LENGTH; ++i) {
                int bit = (ghist >> i) & 1;
                int x_i = bit ? 1 : -1;
                tmp = weights[i + 1] + target * x_i;
                if (tmp > MAX_WEIGHT) tmp = MAX_WEIGHT;
                if (tmp < MIN_WEIGHT) tmp = MIN_WEIGHT;
                weights[i + 1] = tmp;
            }

            // Local history weights
            uint16_t lhist = local_hist[PC];
            for (int i = 0; i < LHIST_LENGTH; ++i) {
                int bit = (lhist >> i) & 1;
                int x_i = bit ? 1 : -1;
                tmp = weights[GHIST_LENGTH + 1 + i] + target * x_i;
                if (tmp > MAX_WEIGHT) tmp = MAX_WEIGHT;
                if (tmp < MIN_WEIGHT) tmp = MIN_WEIGHT;
                weights[GHIST_LENGTH + 1 + i] = tmp;
            }
        }

        // Update local history for this PC
        local_hist[PC] = ((local_hist[PC] << 1) | (taken ? 1 : 0)) & ((1 << LHIST_LENGTH) - 1);
    }
};

// Main Conditional Branch Predictor Class
class SampleCondPredictor {
    SampleHist active_hist;
    std::unordered_map<uint64_t, SampleHist> pred_time_histories;

    PerceptronPredictor perceptron;
    std::unordered_map<uint64_t, int> perceptron_sums;

public:
    SampleCondPredictor() {}

    void setup() {}
    void terminate() {}

    uint64_t get_unique_inst_id(uint64_t seq_no, uint8_t piece) const {
        assert(piece < 16);
        return (seq_no << 4) | (piece & 0x000F);
    }

    bool predict(uint64_t seq_no, uint8_t piece, uint64_t PC, const bool tage_pred) {
        pred_time_histories[get_unique_inst_id(seq_no, piece)] = active_hist;

        return predict_using_given_hist(seq_no, piece, PC, active_hist);
    }

    bool predict_using_given_hist(uint64_t seq_no, uint8_t piece, uint64_t PC, const SampleHist& hist_to_use) {
        bool perc_pred = false;
        int sum = perceptron.compute(hist_to_use.ghist, PC, perc_pred);
        perceptron_sums[get_unique_inst_id(seq_no, piece)] = sum;

        return perc_pred;
    }

    void history_update(uint64_t seq_no, uint8_t piece, uint64_t PC, bool taken, uint64_t nextPC) {
        active_hist.ghist = (active_hist.ghist << 1 | (taken ? 1 : 0)) & ((1ULL << GHIST_LENGTH) - 1);
    }

    void update(uint64_t seq_no, uint8_t piece, uint64_t PC, bool resolveDir, bool predDir, uint64_t nextPC) {
        const auto key = get_unique_inst_id(seq_no, piece);
        const auto& hist = pred_time_histories.at(key);
        const int sum = perceptron_sums.at(key);

        perceptron.update(hist.ghist, PC, resolveDir, sum);

        pred_time_histories.erase(key);
        perceptron_sums.erase(key);
    }

    void update(uint64_t PC, bool resolveDir, bool pred_taken, uint64_t nextPC, const SampleHist& hist_to_use) {
        // not used here
    }
};

#endif

static SampleCondPredictor cond_predictor_impl;
