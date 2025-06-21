import numpy as np
import cv2
import os
import shutil

# --- Configuration ---
NUM_BINS = 200
WIDTH, HEIGHT = 1280, 720
FPS = 30
OUTPUT_FILE = 'p_value_evolution_fast.mp4'
HIST_DATA_FILE = 'histogram_data.bin'

# --- Colors and Fonts ---
BG_COLOR = (255, 255, 255)       # White
BAR_COLOR = (204, 119, 34)       # A nice blue
AXIS_COLOR = (0, 0, 0)           # Black
TEXT_COLOR = (0, 0, 0)           # Black
FONT = cv2.FONT_HERSHEY_SIMPLEX

def draw_histogram_frame(prob_dist, frame_number):
    """Creates a single video frame with a histogram using OpenCV."""
    frame = np.full((HEIGHT, WIDTH, 3), BG_COLOR, dtype=np.uint8)

    # --- Draw Chart Area ---
    x_margin, y_margin = 100, 100
    x_start, y_start = x_margin, HEIGHT - y_margin
    x_end, y_end = WIDTH - x_margin, y_margin
    
    # Draw axes
    cv2.line(frame, (x_start, y_start), (x_end, y_start), AXIS_COLOR, 2) # X-axis
    cv2.line(frame, (x_start, y_start), (x_start, y_end), AXIS_COLOR, 2) # Y-axis

    # --- Draw Bars ---
    bar_width_total = (x_end - x_start) / NUM_BINS
    bar_padding = bar_width_total * 0.1
    bar_width = bar_width_total - bar_padding

    for i, prob in enumerate(prob_dist):
        bar_height = prob * (y_start - y_end)
        
        x1 = int(x_start + i * bar_width_total + bar_padding / 2)
        y1 = int(y_start - bar_height)
        x2 = int(x1 + bar_width)
        y2 = int(y_start)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), BAR_COLOR, -1)

    # --- Draw Text and Labels ---
    cv2.putText(frame, f'P-Value Distribution (Iteration {frame_number})', 
                (x_margin, y_margin - 40), FONT, 1.2, TEXT_COLOR, 2)
    cv2.putText(frame, 'P-Value Bins', 
                (WIDTH // 2 - 100, HEIGHT - 40), FONT, 0.8, TEXT_COLOR, 2)
    cv2.putText(frame, 'Probability', (20, HEIGHT // 2), FONT, 0.8, TEXT_COLOR, 2, cv2.LINE_AA)

    # X-axis labels (0.0, 0.5, 1.0)
    cv2.putText(frame, '0.0', (x_start - 10, y_start + 30), FONT, 0.6, TEXT_COLOR, 1)
    cv2.putText(frame, '0.5', (x_start + (x_end - x_start)//2 - 15, y_start + 30), FONT, 0.6, TEXT_COLOR, 1)
    cv2.putText(frame, '1.0', (x_end - 15, y_start + 30), FONT, 0.6, TEXT_COLOR, 1)
    
    return frame

def create_video_with_opencv(max_frames=None):
    """Reads histogram data and creates a video using OpenCV."""
    if not os.path.exists(HIST_DATA_FILE):
        print(f"Error: Histogram data file not found: '{HIST_DATA_FILE}'")
        return

    # --- Initialize Video Writer ---
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(OUTPUT_FILE, fourcc, FPS, (WIDTH, HEIGHT))

    print(f"Creating video '{OUTPUT_FILE}'...")

    with open(HIST_DATA_FILE, "rb") as f:
        frame_number = 0
        while True:
            if max_frames is not None and frame_number >= max_frames:
                print(f"Reached frame limit of {max_frames}.")
                break

            hist_data = f.read(NUM_BINS * 8)  # 8 bytes for a double
            if not hist_data:
                break
            
            prob_dist = np.frombuffer(hist_data, dtype=np.float64)
            frame = draw_histogram_frame(prob_dist, frame_number)
            video_writer.write(frame)

            if frame_number % 500 == 0:
                print(f"  ...processed frame {frame_number}")
            frame_number += 1
            
    video_writer.release()
    print(f"\nVideo generation complete! Saved to '{os.path.abspath(OUTPUT_FILE)}'")

if __name__ == "__main__":
    create_video_with_opencv(max_frames=1000) 