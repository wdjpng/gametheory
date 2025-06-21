#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <cmath>

class ElFarolSpatial {
private:
    static const int GRID_SIZE = 16;
    static const int NUM_ROUNDS = 5;
    static constexpr double DELTA = 0.002;
    static const int NUM_PLAYERS = GRID_SIZE * GRID_SIZE;
    
    std::vector<double> p_values;
    std::vector<std::vector<double>> payoffs;
    std::vector<std::vector<bool>> attendance_decisions;
    std::vector<double> attendance_history;
    std::vector<double> variance_history;
    std::vector<double> mean_history;
    std::vector<double> p_mean_history;
    std::vector<double> p_median_history;
    
    std::mt19937 rng;
    std::uniform_real_distribution<double> dist;
    std::ofstream p_history_file_bin;

    // Convert 2D grid coordinates to 1D index
    int coord_to_index(int row, int col) const {
        return row * GRID_SIZE + col;
    }
    
    // Convert 1D index to 2D grid coordinates
    std::pair<int, int> index_to_coord(int index) const {
        return {index / GRID_SIZE, index % GRID_SIZE};
    }

    // Calculate payoff for a specific player based on their decision and total attendance
    // Payoff = 1 if: (attended AND bar < 60% full) OR (didn't attend AND bar >= 60% full)
    double calculate_payoff(bool player_attended, int total_attendance) const {
        const int threshold = static_cast<int>(0.6 * NUM_PLAYERS); // 60% of 225 = 135
        bool bar_not_crowded = total_attendance < threshold;
        
        if ((player_attended && bar_not_crowded) || (!player_attended && !bar_not_crowded)) {
            return 1.0;
        }
        return 0.0;
    }

    // Get neighbors of a player (8-connected grid with wraparound)
    std::vector<int> get_neighbors(int player_idx) const {
        auto [row, col] = index_to_coord(player_idx);
        std::vector<int> neighbors;
        
        for (int dr = -1; dr <= 1; ++dr) {
            for (int dc = -1; dc <= 1; ++dc) {
                if (dr == 0 && dc == 0) continue;  // Skip self
                
                int new_row = (row + dr + GRID_SIZE) % GRID_SIZE;
                int new_col = (col + dc + GRID_SIZE) % GRID_SIZE;
                neighbors.push_back(coord_to_index(new_row, new_col));
            }
        }
        return neighbors;
    }

    // Update p-values based on best neighbor performance
    void update_p_values() {
        std::vector<double> new_p_values = p_values;
        
        for (int i = 0; i < NUM_PLAYERS; ++i) {
            auto neighbors = get_neighbors(i);
            
            // Calculate current player's average payoff
            double my_avg_payoff = std::accumulate(payoffs[i].begin(), 
                                                 payoffs[i].end(), 0.0) / NUM_ROUNDS;
            
            // Find neighbor with best average payoff
            double best_avg_payoff = my_avg_payoff;  // Initialize with own payoff
            int best_neighbor = -1;
            
            for (int neighbor : neighbors) {
                double avg_payoff = std::accumulate(payoffs[neighbor].begin(), 
                                                  payoffs[neighbor].end(), 0.0) / NUM_ROUNDS;
                if (avg_payoff > best_avg_payoff) {
                    best_avg_payoff = avg_payoff;
                    best_neighbor = neighbor;
                }
            }
            
            // Only move if a neighbor has better payoff
            if (best_neighbor != -1 && best_avg_payoff > my_avg_payoff) {
                double direction = p_values[best_neighbor] - p_values[i];
                new_p_values[i] += DELTA * direction;
                // Clamp to [0,1]
                new_p_values[i] = std::max(0.0, std::min(1.0, new_p_values[i]));
            }
        }
        
        p_values = new_p_values;
    }

public:
    ElFarolSpatial()
        : p_values(NUM_PLAYERS),
          payoffs(NUM_PLAYERS, std::vector<double>(NUM_ROUNDS)),
          attendance_decisions(NUM_PLAYERS, std::vector<bool>(NUM_ROUNDS)),
          rng(std::random_device{}()),
          dist(0.0, 1.0),
          p_history_file_bin("p_values.bin", std::ios::binary)
    {
        // Initialize p-values with exactly half at 0.2 and half at 0.8, randomly assigned
        std::vector<int> indices(NUM_PLAYERS);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);
        
        for (int i = 0; i < NUM_PLAYERS; ++i) {
            p_values[indices[i]] = (i < NUM_PLAYERS/2) ? 0.95 : 0.25;
        }
        int count_0_2 = std::count_if(p_values.begin(), p_values.end(), 
            [](double p) { return std::abs(p - 0.2) < 1e-6; });
        std::cout << "Number of players with p-value 0.2: " << count_0_2 << std::endl;
    }

    ~ElFarolSpatial() {
        if (p_history_file_bin.is_open()) {
            p_history_file_bin.close();
            std::cout << "P-value history saved to p_values.bin" << std::endl;
        }
    }

    void run_simulation(int n_iterations) {
        for (int iter = 0; iter < n_iterations; ++iter) {
            std::vector<int> round_attendance;
            
            // Play NUM_ROUNDS rounds
            for (int round = 0; round < NUM_ROUNDS; ++round) {
                int attendance = 0;
                
                // Each player decides whether to attend based on their p-value
                for (int i = 0; i < NUM_PLAYERS; ++i) {
                    bool attends = dist(rng) < p_values[i];
                    attendance_decisions[i][round] = attends;
                    if (attends) {
                        attendance++;
                    }
                }
                
                // Calculate and store individual payoffs for this round
                for (int i = 0; i < NUM_PLAYERS; ++i) {
                    payoffs[i][round] = calculate_payoff(attendance_decisions[i][round], attendance);
                }
                
                round_attendance.push_back(attendance);
                attendance_history.push_back(attendance);
            }
            
            // Calculate statistics for this iteration
            double mean = std::accumulate(round_attendance.begin(), 
                                        round_attendance.end(), 0.0) / NUM_ROUNDS;
            double variance = 0.0;
            for (int att : round_attendance) {
                double diff = att - mean;
                variance += diff * diff;
            }
            variance /= NUM_ROUNDS;
            
            mean_history.push_back(mean);
            variance_history.push_back(variance);
            
            // Calculate and store p-value statistics
            double p_mean = std::accumulate(p_values.begin(), p_values.end(), 0.0) / NUM_PLAYERS;
            std::vector<double> sorted_p = p_values;
            std::sort(sorted_p.begin(), sorted_p.end());
            double p_median = sorted_p[NUM_PLAYERS / 2];
            
            p_mean_history.push_back(p_mean);
            p_median_history.push_back(p_median);
            
            // Save current p_values state
            if (p_history_file_bin.is_open()) {
                for (int i = 0; i < NUM_PLAYERS; ++i) {
                    p_history_file_bin.write(reinterpret_cast<const char*>(&iter), sizeof(int));
                    p_history_file_bin.write(reinterpret_cast<const char*>(&p_values[i]), sizeof(double));
                }
            }
            
            // Update p-values based on neighbor performance
            update_p_values();
            
            // Print progress
            if (iter % 1000 == 0) {
                std::cout << "Iteration " << iter << "/" << n_iterations << std::endl;
            }
        }
    }

    void save_results(const std::string& filename) const {
        std::ofstream out(filename);
        out << "iteration,mean_attendance,variance,p_mean,p_median\n";
        for (size_t i = 0; i < mean_history.size(); ++i) {
            out << i << "," << mean_history[i] << "," << variance_history[i] 
                << "," << p_mean_history[i] << "," << p_median_history[i] << "\n";
        }
        out.close();
        std::cout << "Results saved to " << filename << std::endl;
    }
    
    void save_detailed_results(const std::string& filename) const {
        std::ofstream out(filename);
        out << "round,attendance\n";
        for (size_t i = 0; i < attendance_history.size(); ++i) {
            out << i << "," << attendance_history[i] << "\n";
        }
        out.close();
        std::cout << "Detailed results saved to " << filename << std::endl;
    }
    
    void save_final_distribution(const std::string& filename) const {
        std::ofstream out(filename);
        out << "player_id,row,col,p_value\n";
        for (int i = 0; i < NUM_PLAYERS; ++i) {
            auto [row, col] = index_to_coord(i);
            out << i << "," << row << "," << col << "," << p_values[i] << "\n";
        }
        out.close();
        std::cout << "Final distribution saved to " << filename << std::endl;
    }
};

int main() {
    std::cout << "Starting El Farol Spatial Simulation..." << std::endl;
    std::cout << "Grid size: " << 16 << "x" << 16 << " (" << 256 << " players)" << std::endl;
    std::cout << "Rounds per iteration: " << 5 << std::endl;
    std::cout << "Delta: " << 0.002 << std::endl;
    
    ElFarolSpatial simulation;
    simulation.run_simulation(10000);
    simulation.save_results("results.csv");
    simulation.save_detailed_results("detailed_results.csv");
    simulation.save_final_distribution("final_distribution.csv");
    
    std::cout << "Simulation completed!" << std::endl;
    return 0;
} 