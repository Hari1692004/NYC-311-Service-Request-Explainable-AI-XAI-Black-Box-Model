import polars as pl
import glob
from pathlib import Path  # <-- Import the pathlib library
import sys

# --- 1. DEFINE ROBUST PATHS ---
# This assumes your script (this .py file) is in the main project folder
# alongside the '1_raw_data', '2_enrichment_data', etc. folders.
try:
    PROJECT_ROOT = Path(__file__).parent.parent
except NameError:
    # Fallback for interactive environments (like Jupyter)
    PROJECT_ROOT = Path.cwd().parent

# Define all other paths relative to the project root
RAW_DATA_DIR = PROJECT_ROOT / "1_raw_data"
ENRICHMENT_DATA_DIR = PROJECT_ROOT / "2_enrichment_data"
PROCESSED_DATA_DIR = PROJECT_ROOT / "3_processed_data"

# Create the output folder if it doesn't exist
PROCESSED_DATA_DIR.mkdir(exist_ok=True)

# --- 2. DEFINE CONSTANTS ---
# These are the 14 columns we need for the project
REQUIRED_COLUMNS = [
    'Unique Key', 'Created Date', 'Closed Date', 'Agency',
    'Complaint Type', 'Descriptor', 'Incident Zip', 'City',
    'Status', 'Resolution Action Updated Date', 'Community Board',
    'Borough', 'Latitude', 'Longitude'
]

# --- 3. THE INGESTION PLAN (SCHEMA-SAFE & PERFORMANT) ---
print("Starting Layer 1: Ingestion...")

# 1. Get a list of all CSV files using the robust path
all_files = glob.glob(str(RAW_DATA_DIR / "*.csv"))
if not all_files:
    print(f"Error: No CSV files found in {RAW_DATA_DIR}")
    sys.exit(1)

print(f"Found {len(all_files)} files to process in {RAW_DATA_DIR}...")

# 2. Create a list of "lazy" DataFrames
lazy_frames = []
for file_path in all_files:
    # Scan the CSV
    lf_raw = pl.scan_csv(
        file_path,
        ignore_errors=True
    )

    # Get the schema once to fix the performance warning
    schema = lf_raw.collect_schema()

    # Add missing columns as nulls + cast all to String for safety
    lf = lf_raw.with_columns([
        pl.col(col).cast(pl.String).alias(col) if col in schema else pl.lit(None).alias(col)
        for col in REQUIRED_COLUMNS
    ]).select(REQUIRED_COLUMNS) # Enforce order

    lazy_frames.append(lf)

# 3. Combine all individual lazy frames into one big one
lazy_query = pl.concat(lazy_frames)

# --- 4. THE LAZY PROCESSING PLAN (CLEANING & FEATURES) ---
print("Loading enrichment data (us_income_data.csv)...")

# 1. Load your small enrichment (census) data
# We load, clean, and select in one step.
try:
    # First, try to read with latin1 encoding (more permissive)
    try:
        df_census = pl.read_csv(
            ENRICHMENT_DATA_DIR / "us_income_data.csv",
            encoding="latin1",
            separator=",",
            quote_char='"',
            try_parse_dates=False,
            ignore_errors=True,
            infer_schema_length=10000
        )
    except Exception as e:
        print(f"Error with latin1 encoding: {e}")
        # Fall back to reading with specified dtypes
        df_census = pl.read_csv(
            ENRICHMENT_DATA_DIR / "us_income_data.csv",
            encoding="latin1",
            separator=",",
            quote_char='"',
            try_parse_dates=False,
            ignore_errors=True,
            infer_schema_length=0,
            dtypes={
                'Zip_Code': pl.Utf8,
                'Median': pl.Float64,
                'Mean': pl.Float64,
                'sum_w': pl.Float64
            }
        )
    
    # Select and process the columns we need
    df_census = df_census.select([
        # Clean the join key - convert to string and take first 5 digits
        pl.col("Zip_Code").cast(pl.Utf8).str.slice(0, 5).alias("zip_code_join_key"),
        
        # Get the features we want for our model and convert to appropriate types
        pl.col("Median").cast(pl.Float64).alias("Median_Income"),
        pl.col("Mean").cast(pl.Float64).alias("Mean_Income"),
        pl.col("sum_w").cast(pl.Float64).alias("Num_Households")
    ])
    
    # Print the first few rows to verify
    print("Successfully loaded census data. First few rows:")
    print(df_census.head(3))
except FileNotFoundError:
    print(f"Error: us_income_data.csv not found in {ENRICHMENT_DATA_DIR}")
    sys.exit(1)
except Exception as e:
    print(f"Error reading census file. Check column names ('Zip_Code', 'Mean', 'Median', 'sum_w')")
    print(e)
    sys.exit(1)


print("Defining lazy processing query...")

# 2. Define the full cleaning and feature engineering query
# 2. Define the full cleaning and feature engineering query
final_lazy_query = (
    lazy_query
    
    # --- 1. CLEANING & FILTERING ---
    .filter(pl.col("Status") == "Closed")
    
    # --- 2. TYPE CONVERSION ---
    .with_columns([
        # Convert date strings to actual Datetime objects
        pl.col("Created Date").str.strptime(pl.Datetime, format="%m/%d/%Y %I:%M:%S %p", strict=False).alias("Created Date"),
        pl.col("Closed Date").str.strptime(pl.Datetime, format="%m/%d/%Y %I:%M:%S %p", strict=False).alias("Closed Date"),
        pl.col("Resolution Action Updated Date").str.strptime(pl.Datetime, format="%m/%d/%Y %I:%M:%S %p", strict=False).alias("Resolution Action Updated Date"),

        # Convert numeric strings to Floats
        pl.col("Latitude").cast(pl.Float64, strict=False).alias("Latitude"),
        pl.col("Longitude").cast(pl.Float64, strict=False).alias("Longitude")
    ])
    
    # Clean up Incident Zip (remove .0 from floats, trim whitespace)
    .with_columns([
        pl.col("Incident Zip")
        .cast(pl.Utf8)  # Ensure it's a string
        .str.replace(r"\.0*$", "")  # removes float artifacts like "10001.0"
        .str.replace(r"\s+", "")    # remove all whitespace
        .str.slice(0, 5)            # enforce 5 digits
        .alias("Incident Zip")      # maintain column name
    ])

    # --- 3. FEATURE ENGINEERING ---
    .with_columns([
        # Calculate our Target Variable (time to close in hours)
        ((pl.col("Closed Date").cast(pl.Int64) - pl.col("Created Date").cast(pl.Int64)) / 1_000_000 / 3600).alias("Time_to_Close_Hours"),
        
        # Create temporal features
        pl.col("Created Date").dt.weekday().alias("Created_DayOfWeek"),
        pl.col("Created Date").dt.month().alias("Created_Month"),
        pl.col("Created Date").dt.hour().alias("Created_Hour")
    ])
    
    # --- 4. DATA VALIDATION (FILTERING) ---
    .filter(
        (pl.col("Time_to_Close_Hours") > 0.5) &
        (pl.col("Time_to_Close_Hours") < (90 * 24))
    )
    .filter(pl.col("Borough") != "Unspecified")
    .filter(
        pl.col("Latitude").is_not_null() &
        pl.col("Longitude").is_not_null()
    )

    # --- 5. ENRICHMENT ---
    # Join with our census data
    .join(
        df_census.lazy(), 
        left_on="Incident Zip", 
        right_on="zip_code_join_key", 
        how="left"
    )
    
    # --- THIS IS THE FIX ---
    # .drop("zip_code_join_key") # <-- REMOVE THIS LINE
    # --- END FIX ---
)

# --- 5. THE "SINK": EXECUTE AND SAVE TO PARQUET ---
# Define the output path
output_path = PROCESSED_DATA_DIR / "final_features.parquet"

# Run the query and save the file
print(f"Starting ETL process... This may take several minutes.")
print(f"Writing clean data to {output_path}")

try:
    final_lazy_query.sink_parquet(output_path, compression="snappy")
    print("\n-------------------------------------------------")
    print(f"Success! Layer 1 is complete.")
    print(f"Your 'Local Feature Store' is ready at:")
    print(output_path)
    print("-------------------------------------------------")
except Exception as e:
    print(f"\n--- ERROR DURING ETL ---")
    print("The processing pipeline failed. This is often due to an unexpected")
    print("data format. See the error message below for details.")
    print("\nError:")
    print(e)
    print("----------------------------")