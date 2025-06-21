import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_entropy(csv_file='entropy_results.csv', output_image='entropy_plot.png'):
    """
    Reads entropy data from a CSV file and generates a plot.
    """
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: The file '{csv_file}' was not found.")
        print("Please run the data processing script first to generate it.")
        return

    if df.empty:
        print(f"The file '{csv_file}' is empty. No data to plot.")
        return

    print(f"Loaded {len(df)} records from '{csv_file}'")

    # --- Create the Plot ---
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 8))

    sns.lineplot(data=df, x='iteration', y='entropy', color='navy', linewidth=2)

    plt.title('Entropy of P-Value Distribution Over Time', fontsize=18, pad=20)
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Entropy (bits)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    # --- Save and Show Plot ---
    plt.savefig(output_image, dpi=300)
    print(f"Plot saved successfully to '{output_image}'")
    plt.show()

if __name__ == "__main__":
    plot_entropy() 