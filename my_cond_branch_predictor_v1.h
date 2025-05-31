#ifndef _PREDICTOR_H_
#define _PREDICTOR_H_

#include <unordered_map>
#include <vector>
#include <cstdint>
#include <cassert>
#include <cmath>

#define GHIST_LENGTH 32
#define LHIST_LENGTH 8
#define PERCEPTRON_THRESHOLD 75
#define MAX_WEIGHT 127
#define MIN_WEIGHT -128

// Struct for Sample History
struct SampleHist {
    uint64_t ghist;
    bool tage_pred;

    SampleHist() : ghist(0), tage_pred(false) {}
};
int total = 0;
int tage_count = 0;
int prec_count = 0;
// Perceptron Predictor Class
class PerceptronPredictor {
    using Weights = std::vector<int8_t>;

    std::unordered_map<uint64_t, Weights> table;
    std::unordered_map<uint64_t, uint16_t> local_hist; // Per-PC local history

public:
    PerceptronPredictor() {}

    int compute(uint64_t ghist, uint64_t PC, bool& pred) {
        const uint64_t index = (PC >> 1) % 1024;
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
        const uint64_t index = (PC >> 1) % 1024;
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

// Chooser Class for Selecting Between TAGE and Perceptron
class ChooserPredictor {
    using Weights = std::vector<int8_t>;

    // std::unordered_map<uint64_t, Weights> table;  // Perceptron weights for the chooser
    std::unordered_map<uint64_t, uint16_t> chooser_hist; // Per-PC chooser history

public:
    ChooserPredictor() {}
    // static constexpr int GHIST_LENGTH = 16;  // example history length
    static constexpr int MAX_CONFIDENCE = 7;
    static constexpr int MIN_CONFIDENCE = 0;
    static constexpr int CONF_THRESHOLD = 3;
    static constexpr int MAX_WEIGHT_T = 31;
    static constexpr int MIN_WEIGHT_T = -32;
    static constexpr int LEARNING_RATE = 1;
    static constexpr int MARGIN = 0;        // margin for stronger decisions
    static constexpr int DEADZONE = 1;      // zone near 0 where fallback is used
    std::unordered_map<uint64_t, std::vector<int>> table;
    std::unordered_map<uint64_t, int> chooser_conf;

    inline int clamp(int value, int min_val, int max_val) {
        return std::max(min_val, std::min(value, max_val));
    }

    int compute(uint64_t ghist, uint64_t PC, bool tage_pred, bool perc_pred, bool& chooser_pred) {
        const uint64_t index = (PC >> 1) % 1024;
        auto& weights = table[index];

        if (weights.empty()) {
            weights.resize(GHIST_LENGTH + 3);
            for (auto& w : weights) w = (rand() % 3) - 1;  // init weights in [-1, 1]
        }

        int sum = weights[0];  // bias

        // Global history contribution
        for (int i = 0; i < GHIST_LENGTH; ++i) {
            int bit = (ghist >> i) & 1;
            sum += weights[i + 1] * (bit ? 1 : -1);
        }

        // TAGE and perceptron predictor contributions
        sum += weights[GHIST_LENGTH + 1] * (tage_pred ? 1 : -1);
        sum += weights[GHIST_LENGTH + 2] * (perc_pred ? 1 : -1);

        int conf = chooser_conf[index];
        // printf("%d\n", sum);
        // Decision with margin and deadzone fallback
        if (conf >= CONF_THRESHOLD) {
            chooser_pred = (sum >= MARGIN);  // margin-based decision
        } else {
            chooser_pred = true;  // low confidence: use TAGE
        }

        return sum;
    }

    void update(uint64_t ghist, uint64_t PC, bool tage_pred, int perc_sum, bool resolveDir, int sum) {
        const uint64_t index = (PC >> 1) % 1024;
        auto& weights = table[index];

        if (weights.empty()) {
            weights.resize(GHIST_LENGTH + 3);
            for (auto& w : weights) w = (rand() % 3) - 1;
        }

        int target = resolveDir ? 1 : -1;
        bool perc_pred = perc_sum >= 0;

        bool tage_correct = (tage_pred == resolveDir);
        bool perc_correct = (perc_pred == resolveDir);

        int& conf = chooser_conf[index];

        // Chooser picked
        bool chooser_picked_tage = (conf >= CONF_THRESHOLD && sum >= MARGIN) || (conf < CONF_THRESHOLD);

        // Chooser is wrong if it backed the wrong predictor
        bool chooser_was_wrong = (chooser_picked_tage && !tage_correct && perc_correct) ||
                                 (!chooser_picked_tage && tage_correct && !perc_correct);

        
        // Confidence update
        if (chooser_was_wrong) {
            if (conf > MIN_CONFIDENCE) conf--;
        } else {
            if (conf < MAX_CONFIDENCE) conf++;
        }

        // Train chooser if one predictor is correct and the other is not
        if (tage_correct != perc_correct) {
            int correct_target = tage_correct ? 1 : -1;

            // Bias term (with tighter clamping for stability)
            weights[0] = clamp(weights[0] + LEARNING_RATE * correct_target, -8, 8);

            // Global history
            for (int i = 0; i < GHIST_LENGTH; ++i) {
                int bit = (ghist >> i) & 1;
                int x_i = bit ? 1 : -1;
                weights[i + 1] = clamp(weights[i + 1] + LEARNING_RATE * correct_target * x_i, MIN_WEIGHT_T, MAX_WEIGHT_T);
            }

            // TAGE/Perceptron predictor contributions
            weights[GHIST_LENGTH + 1] = clamp(0.97*weights[GHIST_LENGTH + 1] + LEARNING_RATE * correct_target * (tage_pred ? 1 : -1), MIN_WEIGHT_T, MAX_WEIGHT_T);
            weights[GHIST_LENGTH + 2] = clamp(0.97*weights[GHIST_LENGTH + 2] + LEARNING_RATE * correct_target * (perc_pred ? 1 : -1), MIN_WEIGHT_T, MAX_WEIGHT_T);
        }
    
        // Update chooser history (optional, useful if you're using it elsewhere)
        // chooser_hist[PC] = ((chooser_hist[PC] << 1) | (resolveDir ? 1 : 0)) & ((1 << GHIST_LENGTH) - 1);
    }    
    
};
std::unordered_map<uint64_t, bool> tage_prediction;

// Main Conditional Branch Predictor Class
class SampleCondPredictor {
    SampleHist active_hist;
    std::unordered_map<uint64_t, SampleHist> pred_time_histories;

    PerceptronPredictor perceptron;
    ChooserPredictor chooser;
    std::unordered_map<uint64_t, int> perceptron_sums;
    std::unordered_map<uint64_t, int> chooser_sums;
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
        pred_time_histories[get_unique_inst_id(seq_no, piece)] = active_hist;

        return predict_using_given_hist(seq_no, piece, PC, active_hist, true);
    }

    bool predict_using_given_hist(uint64_t seq_no, uint8_t piece, uint64_t PC, const SampleHist& hist_to_use, const bool pred_time_predict) {
        bool perc_pred = false;
        int sum = perceptron.compute(hist_to_use.ghist, PC, perc_pred);
        perceptron_sums[get_unique_inst_id(seq_no, piece)] = sum;
        tage_prediction[get_unique_inst_id(seq_no, piece)] = hist_to_use.tage_pred;
        // Use chooser to decide between TAGE and Perceptron predictions
        bool chooser_pred = false;
        int chooser_sum = chooser.compute(hist_to_use.ghist, PC, hist_to_use.tage_pred, perc_pred, chooser_pred);
        
        chooser_sums[get_unique_inst_id(seq_no, piece)] = chooser_sum;
        return chooser_pred ? hist_to_use.tage_pred : perc_pred;
    }

    void history_update(uint64_t seq_no, uint8_t piece, uint64_t PC, bool taken, uint64_t nextPC) {
        active_hist.ghist = (active_hist.ghist << 1 | (taken ? 1 : 0));
    }

    void update(uint64_t seq_no, uint8_t piece, uint64_t PC, bool resolveDir, bool predDir, uint64_t nextPC) {
        const auto key = get_unique_inst_id(seq_no, piece);
        const auto& hist = pred_time_histories.at(key);
        const int sum = perceptron_sums.at(key);
        bool tage_pred = tage_prediction.at(key);
        const int chooser_sum = chooser_sums.at(key);
        
        
        /**
        if(tage_pred != (sum > 0)) {
            // printf("Total %d\n", total++);
            if(chooser_sum > 0) {
                if(resolveDir != tage_pred) {
                    printf("TAGE %d\n", tage_count++);
                }
            }
            else {
                if(resolveDir != (sum > 0)) {
                    // printf("Prec %d\n", prec_count++);
                }
            }
        }
        */

        /**
        if(1) {
            // printf("Total %d\n", total++);
            if(chooser_sum > 0) {
                if(1) {
                    printf("TAGE %d\n", tage_count++);
                }
            }
            else {
                if(1) {
                    // printf("Prec %d\n", prec_count++);
                }
            }
        }
        */

        perceptron.update(hist.ghist, PC, resolveDir, sum);

        // Update chooser
        chooser.update(hist.ghist, PC, hist.tage_pred, sum, resolveDir, chooser_sum);

        pred_time_histories.erase(key);
        perceptron_sums.erase(key);
    }

    void update(uint64_t PC, bool resolveDir, bool pred_taken, uint64_t nextPC, const SampleHist& hist_to_use) {
        // not used here
    }
};

#endif

static SampleCondPredictor cond_predictor_impl;
