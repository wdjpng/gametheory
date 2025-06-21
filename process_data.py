import numpy as np
import struct
import sys

NUM_BINS = 200
NUM_PLAYERS = 256
RECORD_SIZE = 12  # sizeof(int) + sizeof(double)
CHUNK_SIZE = NUM_PLAYERS * RECORD_SIZE * 100 # Process 100 iterations at a time

def process_binary_data_python():
    print("Processing data with Python script...")
    
    with open("p_values.bin", "rb") as f_in, \
         open("entropy_results.csv", "w") as f_entropy, \
         open("histogram_data.bin", "wb") as f_hist:
        
        f_entropy.write("iteration,entropy\n")
        
        last_iteration = -1
        p_values = []

        while True:
            chunk = f_in.read(CHUNK_SIZE)
            if not chunk:
                break

            for i in range(0, len(chunk), RECORD_SIZE):
                record_data = chunk[i:i+RECORD_SIZE]
                if len(record_data) < RECORD_SIZE:
                    continue
                
                iteration, p_value = struct.unpack("<id", record_data)

                if last_iteration != -1 and iteration != last_iteration:
                    # Process completed iteration
                    hist, _ = np.histogram(p_values, bins=NUM_BINS, range=(0, 1))
                    prob_dist = hist / np.sum(hist)
                    
                    entropy = -np.sum(prob_dist[prob_dist > 0] * np.log2(prob_dist[prob_dist > 0]))
                    
                    f_entropy.write(f"{last_iteration},{entropy}\n")
                    
                    prob_dist_double = np.array(prob_dist, dtype=np.float64)
                    f_hist.write(prob_dist_double.tobytes())

                    if last_iteration % 1000 == 0:
                        print(f"Processed iteration {last_iteration}")
                        
                    p_values = []

                p_values.append(p_value)
                last_iteration = iteration

        # Process the very last iteration
        if p_values:
            hist, _ = np.histogram(p_values, bins=NUM_BINS, range=(0, 1))
            prob_dist = hist / np.sum(hist)
            entropy = -np.sum(prob_dist[prob_dist > 0] * np.log2(prob_dist[prob_dist > 0]))
            f_entropy.write(f"{last_iteration},{entropy}\n")
            
            prob_dist_double = np.array(prob_dist, dtype=np.float64)
            f_hist.write(prob_dist_double.tobytes())
            if last_iteration % 1000 == 0:
                print(f"Processed iteration {last_iteration}")


    print("Processing complete.")
    print("Entropy data saved to entropy_results.csv")
    print("Histogram data for video saved to histogram_data.bin")


if __name__ == "__main__":
    process_binary_data_python() 