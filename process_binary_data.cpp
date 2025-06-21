#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <cmath>
#include <numeric>
#include <algorithm>

const int NUM_BINS = 20;
const int NUM_PLAYERS = 256;

struct Record {
    int iteration;
    double p_value;
};

int main() {
    std::ifstream input_file("p_values.bin", std::ios::binary);
    if (!input_file) {
        std::cerr << "Error opening p_values.bin" << std::endl;
        return 1;
    }

    std::ofstream entropy_file("entropy_results.csv");
    entropy_file << "iteration,entropy" << std::endl;

    std::ofstream hist_file("histogram_data.bin", std::ios::binary);

    Record rec;
    input_file.read(reinterpret_cast<char*>(&rec), sizeof(Record));
    if (input_file.gcount() == 0) {
        std::cout << "Input file is empty." << std::endl;
        return 0;
    }

    do {
        int current_iteration = rec.iteration;
        std::vector<double> p_values;
        
        do {
            p_values.push_back(rec.p_value);
            input_file.read(reinterpret_cast<char*>(&rec), sizeof(Record));
        } while (input_file && rec.iteration == current_iteration);

        std::vector<int> hist(NUM_BINS, 0);
        for (double p : p_values) {
            int bin = std::min(static_cast<int>(p * NUM_BINS), NUM_BINS - 1);
            hist[bin]++;
        }

        std::vector<double> prob_dist(NUM_BINS, 0.0);
        if (!p_values.empty()) {
            for(size_t i = 0; i < hist.size(); ++i) {
                prob_dist[i] = static_cast<double>(hist[i]) / p_values.size();
            }
        }
        
        double entropy = 0.0;
        for (double p : prob_dist) {
            if (p > 0) {
                entropy -= p * std::log2(p);
            }
        }

        entropy_file << current_iteration << "," << entropy << std::endl;
        hist_file.write(reinterpret_cast<const char*>(prob_dist.data()), NUM_BINS * sizeof(double));
        
        if (current_iteration % 1000 == 0) {
            std::cout << "Processed iteration " << current_iteration << std::endl;
        }

    } while (input_file);

    std::cout << "Processing complete." << std::endl;
    std::cout << "Entropy data saved to entropy_results.csv" << std::endl;
    std::cout << "Histogram data for video saved to histogram_data.bin" << std::endl;

    return 0;
} 