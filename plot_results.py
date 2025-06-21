import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_results():
    """Plot variance and mean attendance over time from simulation results."""
    
    # Set style for better-looking plots
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    try:
        # Load the results
        df = pd.read_csv('results.csv')
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('El Farol Spatial Simulation Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Mean attendance over time
        ax1.plot(df['iteration'], df['mean_attendance'], 'b-', linewidth=2, alpha=0.8)
        ax1.axhline(y=154, color='r', linestyle='--', alpha=0.7, label='Optimal attendance (154)')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Mean Attendance')
        ax1.set_title('Mean Attendance Over Time')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Variance over time
        ax2.plot(df['iteration'], df['variance'], 'g-', linewidth=2, alpha=0.8)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Variance')
        ax2.set_title('Attendance Variance Over Time')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Mean and Median p-values over time
        ax3.plot(df['iteration'], df['p_mean'], 'b-', linewidth=2, alpha=0.8, label='Mean p-value')
        ax3.plot(df['iteration'], df['p_median'], 'r-', linewidth=2, alpha=0.8, label='Median p-value')
        ax3.axhline(y=0.6, color='g', linestyle='--', alpha=0.7, label='Optimal p-value (0.6)')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('p-value')
        ax3.set_title('Mean and Median p-values Over Time')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Plot 4: Both metrics on same plot (normalized)
        # Normalize both metrics to [0,1] for comparison
        mean_norm = (df['mean_attendance'] - df['mean_attendance'].min()) / (df['mean_attendance'].max() - df['mean_attendance'].min())
        var_norm = (df['variance'] - df['variance'].min()) / (df['variance'].max() - df['variance'].min())
        
        ax4.plot(df['iteration'], mean_norm, 'b-', linewidth=2, alpha=0.8, label='Mean Attendance (normalized)')
        ax4.plot(df['iteration'], var_norm, 'g-', linewidth=2, alpha=0.8, label='Variance (normalized)')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Normalized Value')
        ax4.set_title('Normalized Mean Attendance and Variance')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig('simulation_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary statistics
        print("\n=== Simulation Summary ===")
        print(f"Total iterations: {len(df)}")
        print(f"Mean attendance - Final: {df['mean_attendance'].iloc[-1]:.2f}, Average: {df['mean_attendance'].mean():.2f}")
        print(f"Variance - Final: {df['variance'].iloc[-1]:.2f}, Average: {df['variance'].mean():.2f}")
        print(f"Mean p-value - Final: {df['p_mean'].iloc[-1]:.4f}, Average: {df['p_mean'].mean():.4f}")
        print(f"Median p-value - Final: {df['p_median'].iloc[-1]:.4f}, Average: {df['p_median'].mean():.4f}")
        print(f"Optimal attendance (135) achieved in {(df['mean_attendance'] < 135.5).sum()} iterations")
        
        # Load and plot detailed results if available
        try:
            detailed_df = pd.read_csv('detailed_results.csv')
            
            plt.figure(figsize=(15, 6))
            plt.plot(detailed_df['round'], detailed_df['attendance'], 'b-', alpha=0.6, linewidth=1)
            plt.axhline(y=135, color='r', linestyle='--', alpha=0.7, label='Optimal attendance (135)')
            plt.xlabel('Round')
            plt.ylabel('Attendance')
            plt.title('Detailed Attendance Over All Rounds')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Add moving average
            window = 25
            if len(detailed_df) > window:
                moving_avg = detailed_df['attendance'].rolling(window=window).mean()
                plt.plot(detailed_df['round'], moving_avg, 'r-', linewidth=2, alpha=0.8, 
                        label=f'Moving Average ({window} rounds)')
                plt.legend()
            
            plt.tight_layout()
            plt.savefig('detailed_attendance.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        except FileNotFoundError:
            print("Detailed results file not found. Skipping detailed plot.")
        
        # Plot final distribution
        try:
            dist_df = pd.read_csv('final_distribution.csv')
            
            # Create distribution plots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Final Distribution of Player Strategies (p-values)', fontsize=16, fontweight='bold')
            
            # Histogram of p-values
            ax1.hist(dist_df['p_value'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.axvline(x=dist_df['p_value'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {dist_df["p_value"].mean():.3f}')
            ax1.axvline(x=0.6, color='green', linestyle='--', alpha=0.7, 
                       label='Optimal (0.6)')
            ax1.set_xlabel('p-value (probability of attending)')
            ax1.set_ylabel('Number of players')
            ax1.set_title('Distribution of p-values')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Spatial heatmap
            p_grid = dist_df.pivot(index='row', columns='col', values='p_value')
            im = ax2.imshow(p_grid, cmap='viridis', aspect='equal')
            ax2.set_title('Spatial Distribution of p-values')
            ax2.set_xlabel('Column')
            ax2.set_ylabel('Row')
            plt.colorbar(im, ax=ax2, label='p-value')
            
            # Box plot by row
            row_data = [dist_df[dist_df['row'] == r]['p_value'].values for r in range(15)]
            ax3.boxplot(row_data, labels=range(15))
            ax3.set_xlabel('Row')
            ax3.set_ylabel('p-value')
            ax3.set_title('p-value Distribution by Row')
            ax3.grid(True, alpha=0.3)
            
            # Box plot by column
            col_data = [dist_df[dist_df['col'] == c]['p_value'].values for c in range(15)]
            ax4.boxplot(col_data, labels=range(15))
            ax4.set_xlabel('Column')
            ax4.set_ylabel('p-value')
            ax4.set_title('p-value Distribution by Column')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('final_distribution.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Print distribution statistics
            print(f"\n=== Final Distribution Statistics ===")
            print(f"Mean p-value: {dist_df['p_value'].mean():.4f}")
            print(f"Std p-value: {dist_df['p_value'].std():.4f}")
            print(f"Min p-value: {dist_df['p_value'].min():.4f}")
            print(f"Max p-value: {dist_df['p_value'].max():.4f}")
            print(f"Expected attendance: {dist_df['p_value'].mean() * 225:.1f} players")
            
        except FileNotFoundError:
            print("Final distribution file not found. Skipping distribution plots.")
            
    except FileNotFoundError:
        print("Results file 'results.csv' not found. Please run the C++ simulation first.")
        return
    except Exception as e:
        print(f"Error plotting results: {e}")
        return

if __name__ == "__main__":
    plot_results() 