
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
from pathlib import Path

# --- Setup Paths ---
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = BASE_DIR / "data/Processed Data/final_model_data.csv"
PREDICTIONS_PATH = BASE_DIR / "code/RF/outputs/test_predictions.csv"

def main():
    print("🚀 Loading data for Analysis...")
    
    # ---------------------------------------------------------
    # PART 1: LAG BEHAVIOR (Autocorrelation)
    # ---------------------------------------------------------
    print("📈 Generating Autocorrelation Plot...")
    df_raw = pd.read_csv(RAW_DATA_PATH)
    
    plt.figure(figsize=(12, 5))
    # Look back 200 hours to capture the 24h and 168h cycles
    plot_acf(df_raw["ND"].dropna(), lags=200, alpha=0.05, title="Autocorrelation of National Demand (Lag Behavior)")
    plt.xlabel("Lag (Hours)")
    plt.ylabel("Correlation Coefficient")

    # Highlight the specific lags you engineered
    plt.axvline(x=24, color='r', linestyle='--', alpha=0.7, label='24h (Daily Routine)')
    plt.axvline(x=168, color='g', linestyle='--', alpha=0.7, label='168h (Weekly Routine)')

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show() # Close this window to move to the next plots

    # ---------------------------------------------------------
    # PART 2: VISUAL ERROR ANALYSIS
    # ---------------------------------------------------------
    print("🔍 Generating Visual Error Analysis...")
    
    # Load the predictions saved by your tuning script
    try:
        results_df = pd.read_csv(PREDICTIONS_PATH)
    except FileNotFoundError:
        print(f"❌ Could not find {PREDICTIONS_PATH}. Make sure your tuning script ran completely!")
        return

    # Convert datetime back to timestamp objects
    results_df['datetime'] = pd.to_datetime(results_df['datetime'])
    
    # Calculate Absolute Error
    results_df['Absolute_Error'] = abs(results_df['actual_ND'] - results_df['predicted_ND'])

    # Extract time features for plotting
    results_df['hour'] = results_df['datetime'].dt.hour
    results_df['dayofweek'] = results_df['datetime'].dt.dayofweek # 0=Mon, 6=Sun

    # Setup the canvas for two side-by-side bar charts
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot A: Error by Hour
    sns.barplot(data=results_df, x='hour', y='Absolute_Error', ax=axes[0], color='#1f77b4')
    axes[0].set_title('Average Forecasting Error by Hour of Day', fontweight='bold')
    axes[0].set_xlabel('Hour of Day (0 = Midnight)')
    axes[0].set_ylabel('Absolute Error (MW)')

    # Plot B: Error by Day of Week
    sns.barplot(data=results_df, x='dayofweek', y='Absolute_Error', ax=axes[1], color='#ff7f0e')
    axes[1].set_title('Average Forecasting Error by Day of Week', fontweight='bold')
    axes[1].set_xlabel('Day of Week (0 = Monday, 6 = Sunday)')
    axes[1].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    axes[1].set_ylabel('Absolute Error (MW)')

    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # PART 3: FIND THE WORST PREDICTION
    # ---------------------------------------------------------
    worst_idx = results_df['Absolute_Error'].idxmax()
    worst_day = results_df.loc[worst_idx]
    
    print(f"\n🚨 WORST SINGLE PREDICTION OF 2025:")
    print(f"Time: {worst_day['datetime']}")
    print(f"Actual: {round(worst_day['actual_ND'], 1)} MW")
    print(f"Predicted: {round(worst_day['predicted_ND'], 1)} MW")
    print(f"Error Magnitude: {round(worst_day['Absolute_Error'], 1)} MW")

if __name__ == "__main__":
    main()