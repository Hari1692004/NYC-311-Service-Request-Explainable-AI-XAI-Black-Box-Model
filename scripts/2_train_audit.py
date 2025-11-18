import polars as pl
import lightgbm as lgbm
import shap
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split
import gc # Import garbage collector

# --- 1. DEFINE ROBUST PATHS ---
# (This block is unchanged)
try:
    PROJECT_ROOT = Path(__file__).parent.parent
except NameError:
    PROJECT_ROOT = Path.cwd()

PROCESSED_DATA_DIR = PROJECT_ROOT / "3_processed_data"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "4_model_outputs"
MODEL_OUTPUT_DIR.mkdir(exist_ok=True)

# --- 2. DEFINE CONSTANTS & CONFIGURATION ---
# (This block is unchanged)
PARQUET_PATH = PROCESSED_DATA_DIR / "final_features.parquet"
MODEL_PATH = MODEL_OUTPUT_DIR / "lgbm_model.joblib"
AUDIT_PLOT_PATH = MODEL_OUTPUT_DIR / "shap_summary_plot.png"
SAMPLE_SIZE = 250_000
TARGET_VARIABLE = "Time_to_Close_Hours"
CATEGORICAL_FEATURES = [
    'Agency', 'Complaint Type', 'Descriptor', 'City', 
    'Community Board', 'Borough'
]
NUMERIC_FEATURES = [
    'Latitude', 'Longitude', 'Created_DayOfWeek', 'Created_Month',
    'Created_Hour', 'Median_Income', 'Mean_Income', 'Num_Households'
]

# --- 3. OPTIMIZED DATA LOADING ---

def load_data(parquet_path, sample_size):
    """
    OPTIMIZED: Uses lazy scanning to load only a sample
    into memory, instead of the full 3.5 GB file.
    """
    print(f"Lazily scanning {parquet_path}...")
    try:
        # 1. SCAN the file (no data is loaded)
        lazy_df = pl.scan_parquet(parquet_path)
    except FileNotFoundError:
        print(f"--- ERROR ---")
        print(f"File not found: {parquet_path}")
        print("Please run the '1_process_raw_data.py' script first.")
        sys.exit(1)
    
    # 2. Get the total number of rows from the scan plan
    total_rows = lazy_df.select(pl.count()).collect().item()
    
    # 3. Take a sample.
    if total_rows > sample_size:
        print(f"File has {total_rows:,} rows. Taking a random sample of {sample_size:,}...")
        # First collect the data, then sample
        df = lazy_df.collect().sample(n=sample_size, shuffle=True, seed=42)
    else:
        print("File is smaller than sample size. Loading all data.")
        df = lazy_df.collect()
    
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df

# --- 4. NOTEBOOK HYGIENE ---
def run_notebook_hygiene(df_to_delete, X_to_delete):
    """
    In a notebook, we must manually delete large DataFrames
    from memory to make room for the next step.
    """
    print("Running garbage collection to free up memory...")
    try:
        del df_to_delete
        del X_to_delete
    except:
        pass # Ignore errors if they don't exist
    
    gc.collect()
    print("Memory freed.")


# --- 5. MODELING & AUDIT FUNCTIONS ---
#
# The following functions are UNCHANGED.
# They are already well-optimized for memory.
#
def prepare_features(df, numeric_cols, cat_cols, target_col):
    print("Preparing features for modeling...")
    df = df.with_columns(
        [pl.col(c).fill_null("MISSING").cast(pl.Categorical) for c in cat_cols]
    )
    X = df.select(numeric_cols + cat_cols).to_pandas()
    y = df.select(target_col).to_pandas().squeeze()
    print("Features prepared.")
    return X, y

def train_model(X_train, y_train, X_test, y_test, cat_features):
    print("Training LightGBM model...")
    model = lgbm.LGBMRegressor(
        objective='regression_l1',
        n_estimators=1000,
        n_jobs=-1,
        random_state=42
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='mae',
        callbacks=[lgbm.early_stopping(50, verbose=True)],
        categorical_feature=cat_features
    )
    print("Model training complete.")
    return model

def save_model(model, path):
    print(f"Saving model to {path}...")
    joblib.dump(model, path)
    print("Model saved.")

def run_shap_audit(model, X_test, output_plot_path):
    print("Starting SHAP audit... (This may take a few minutes)")
    
    # This sample is the key to SHAP not crashing.
    X_test_sample = shap.sample(X_test, 1000, random_state=42)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_sample)
    
    print(f"Saving SHAP summary plot to {output_plot_path}...")
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_test_sample,
        show=False,
        feature_names=X_test.columns
    )
    plt.title("SHAP Feature Impact on Ticket Resolution Time")
    plt.savefig(output_plot_path, bbox_inches='tight')
    plt.close()
    
    print("SHAP plot saved.")

# --- 6. MAIN EXECUTION ---

def main():
    """Main execution pipeline for Layer 2."""
    print("--- Starting Layer 2: Model Training & Audit ---")
    
    df = load_data(PARQUET_PATH, SAMPLE_SIZE)
    X, y = prepare_features(df, NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET_VARIABLE)
    
    # --- HYGIENE STEP ---
    # We've created X and y, so we no longer need the 
    # massive 'df' DataFrame. Let's delete it.
    run_notebook_hygiene(df, None) 
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- HYGIENE STEP ---
    # We've created train/test splits, so we no longer
    # need the full 'X' and 'y' objects.
    run_notebook_hygiene(None, X) 
    
    model = train_model(X_train, y_train, X_test, y_test, CATEGORICAL_FEATURES)
    save_model(model, MODEL_PATH)
    
    # The SHAP audit is already optimized to use a small sample (X_test_sample)
    # so we don't need to do any more hygiene.
    run_shap_audit(model, X_test, AUDIT_PLOT_PATH)
    
    print("\n-------------------------------------------------")
    print(f"Success! Layer 2 is complete.")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Audit plot saved to: {AUDIT_PLOT_PATH}")
    print("-------------------------------------------------")
    print("\nNext step: Run '3_run_dashboard.py' to see your results!")

if __name__ == "__main__":
    main()
