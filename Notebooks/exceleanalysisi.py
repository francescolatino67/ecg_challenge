import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import glob

def preprocess_data(df):
    """
    Cleans the DataFrame by removing specific columns and performing One-Hot Encoding.
    """
    # Columns to remove
    # NOTE: ECG_patient_id is KEPT here for later balancing, and removed afterwards
    cols_to_drop = [
        'code', 'AV block', 'ST abnormality', 
        'Complete BBB', 'Prolonged QTc', 'Supraventricular arrhythmias', 
        'Ventricular arrhythmias', 'Baseline ECG abnormalities'
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Fix typo in column name if present
    if 'trainning_load' in df.columns:
        df = df.rename(columns={'trainning_load': 'training_load'})

    # One-Hot Encoding for 'training_load'
    if 'training_load' in df.columns:
        # Convert to numeric, handle NaNs, and convert to integer
        # We fill NaNs with -1 so they don't interfere with 0-4 classes, then cast to int
        temp_load = pd.to_numeric(df['training_load'], errors='coerce').fillna(-1).astype(int)
        
        # Create dummies
        dummies = pd.get_dummies(temp_load, prefix='training_load')
        
        # Ensure we have columns for 0, 1, 2, 3, 4
        required_classes = [0, 1, 2, 3, 4]
        selected_cols = []
        
        for i in required_classes:
            col_name = f'training_load_{i}'
            if col_name not in dummies.columns:
                dummies[col_name] = 0
            # Convert to int (0/1) instead of bool
            dummies[col_name] = dummies[col_name].astype(int)
            selected_cols.append(col_name)
        
        # Select only the columns we want in the specific order
        dummies = dummies[selected_cols]
        
        # Concatenate and drop original
        df = pd.concat([df.drop(columns=['training_load']), dummies], axis=1)
    
    return df

def visualize_comparison(v1, v2, name1, name2, col_name):
    """
    Creates a self-contained figure comparing two distributions and their total.
    Uses absolute counts (histplot).
    """
    plt.figure(figsize=(12, 7))
    
    # Combined data for the "Total" curve
    v_total = pd.concat([v1, v2])

    # Use histplot with element="step" for clear overlapping of counts
    sns.histplot(v1, label=f"Batch 1: {name1}", color="blue", element="step", stat="count", alpha=0.3)
    sns.histplot(v2, label=f"Batch 2: {name2}", color="orange", element="step", stat="count", alpha=0.3)
    sns.histplot(v_total, label="Total (Batch 1 + Batch 2)", color="black", element="step", fill=False, stat="count", linewidth=2)
    
    plt.title(f"Comparison of {col_name} (Absolute Counts)")
    plt.xlabel(col_name)
    plt.ylabel("Absolute Count")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
    plt.close()

def visualize_sport_classification(df, col_name, target_col='sport_ability'):
    """
    Visualizes the distribution of a column split by sport_classification (0 vs 1).
    """
    plt.figure(figsize=(10, 6))
    
    # Check if target column exists
    if target_col not in df.columns:
        return

    # Use histplot with multiple="stack" or "dodge"
    # "stack" shows the total count and how it's divided
    # "dodge" shows them side-by-side
    sns.histplot(data=df, x=col_name, hue=target_col, multiple="stack", element="step", palette="viridis")
    
    plt.title(f"Distribution of {col_name} by {target_col}")
    plt.xlabel(col_name)
    plt.ylabel("Count")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
    plt.close()

def balance_dataset(df, target_col='sport_ability', id_col='ECG_patient_id', random_state=42):
    """
    Balances the dataset by undersampling the majority class in 'target_col'.
    Returns:
        balanced_df (pd.DataFrame): The balanced dataset (with id_col REMOVED).
        removed_ids (list): List of IDs from the removed rows.
    """
    if target_col not in df.columns:
        print(f"Warning: {target_col} not found. Returning original df.")
        return df, []
    
    if id_col not in df.columns:
        print(f"Warning: {id_col} not found. Returning original df.")
        return df, []

    # Identify classes
    counts = df[target_col].value_counts()
    if len(counts) < 2:
        print("Warning: Only one class present. Cannot balance.")
        return df, []
        
    minority_class = counts.idxmin()
    majority_class = counts.idxmax()
    
    n_minority = counts[minority_class]
    
    df_minority = df[df[target_col] == minority_class]
    df_majority = df[df[target_col] == majority_class]
    
    # Undersample majority
    df_majority_downsampled = df_majority.sample(n=n_minority, random_state=random_state)
    
    # Identify removed rows
    # We use the index to find which rows from df_majority were NOT selected
    removed_indices = df_majority.index.difference(df_majority_downsampled.index)
    removed_rows = df_majority.loc[removed_indices]
    removed_ids = removed_rows[id_col].tolist()
    
    # Combine
    balanced_df = pd.concat([df_majority_downsampled, df_minority])
    
    # Shuffle the result so classes aren't grouped
    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Drop the ID column from the final dataset as requested
    balanced_df = balanced_df.drop(columns=[id_col])
    print (balanced_df)
    
    return balanced_df, removed_ids

def analyze_excel_files(data_dir):
    # Find all Excel files in the Data directory, filtering out temporary ones (~$...)
    excel_files = [f for f in glob.glob(os.path.join(data_dir, "*.xlsx")) if not os.path.basename(f).startswith("~$")]
    
    if len(excel_files) < 2:
        print(f"Need at least two Excel files for comparison. Found: {len(excel_files)}")
        return

    # Take the first two files found
    file1_path, file2_path = excel_files[0], excel_files[1]
    name1, name2 = os.path.basename(file1_path), os.path.basename(file2_path)

    try:
        print(f"Loading {name1}...")
        df1 = pd.read_excel(file1_path, engine='openpyxl')
        print(f"Loading {name2}...")
        df2 = pd.read_excel(file2_path, engine='openpyxl')

        # Preprocess both files
        print("Preprocessing DataFrames...")
        
        # Keep raw copies for the final inclusive analysis
        df1_raw = df1.copy()
        df2_raw = df2.copy()
        
        df1 = preprocess_data(df1)
        df2 = preprocess_data(df2)

        # Individual Analysis (Zero Rows/Cols) for reporting
        for df, name in [(df1, name1), (df2, name2)]:
            print(f"\n{'='*50}")
            print(f"Zero-Value Analysis: {name}")
            print(f"{'='*50}")
            zero_cols = df.columns[(df == 0).all()]
            print(f"Columns with all zeros ({len(zero_cols)}):")
            if len(zero_cols) > 0: print(zero_cols.tolist())
            else: print("None found.")
            
            numeric_df = df.select_dtypes(include=['number'])
            zero_rows = numeric_df[(numeric_df == 0).all(axis=1)]
            print(f"Rows with all zeros (numeric columns only) ({len(zero_rows)}):")
            if len(zero_rows) > 0: print(f"Indices: {zero_rows.index.tolist()}")
            else: print("None found.")

        # Combined Distributions
        print(f"\n{'='*50}")
        print(f"Comparing Distributions: {name1} vs {name2}")
        print(f"{'='*50}")
        
        cols1 = df1.select_dtypes(include=['number']).columns
        cols2 = df2.select_dtypes(include=['number']).columns
        common_cols = cols1.intersection(cols2)

        if not common_cols.empty:
            for col in common_cols:
                visualize_comparison(df1[col].dropna(), df2[col].dropna(), name1, name2, col)
        else:
            print("No common numeric columns found to compare.")

        # --- DATASET BALANCING ---
        print(f"\n{'='*50}")
        print(f"Dataset Balancing (Target: sport_ability)")
        print(f"{'='*50}")
        
        # Balance df1
        print(f"Balancing {name1}...")
        df1_balanced, removed_ids1 = balance_dataset(df1)
        print(f"Removed {len(removed_ids1)} patients from {name1}.")
        # print(f"Sample removed IDs: {removed_ids1[:5]} ...")
        
        # Balance df2
        print(f"Balancing {name2}...")
        df2_balanced, removed_ids2 = balance_dataset(df2)
        print(f"Removed {len(removed_ids2)} patients from {name2}.")
        
        # Verify Balanced Dataset
        print(f"\n{'='*50}")
        print(f"Verification of BALANCED Datasets")
        print(f"{'='*50}")
        
        for df_b, name in [(df1_balanced, name1), (df2_balanced, name2)]:
            print(f"-- {name} Balanced --")
            print(f"Shape: {df_b.shape}")
            if 'sport_ability' in df_b.columns:
                print(f"sport_ability Distribution:\n{df_b['sport_ability'].value_counts()}")
            print(f"Columns: {list(df_b.columns)}")
            print("First 10 rows:")
            print(df_b.head(10))
            print("-" * 30)

        # Sport Ability Analysis (Combined Data - ALL COLUMNS)
        print(f"\n{'='*50}")
        print(f"Sport Ability Analysis (Combined Data - All Columns)")
        print(f"{'='*50}")
        
        # Combine RAW dataframes for this analysis to include dropped columns
        df_total_raw = pd.concat([df1_raw, df2_raw], ignore_index=True)
        
        target_col = 'sport_ability'
        if target_col in df_total_raw.columns:
            # Analyze against all numeric columns in the RAW data
            numeric_cols = df_total_raw.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                if col != target_col:
                    visualize_sport_classification(df_total_raw, col, target_col)
        else:
            print(f"Column '{target_col}' not found in the combined dataset.")

    except PermissionError:
        print("\nERROR: Permission denied. Please close any open Excel files and try again.")
    except Exception as e:
        import traceback
        print(f"Error during analysis: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    data_folder = r"C:\Users\Marco\OneDrive - Politecnico di Milano\Desktop\Dottorato\Corsi\AI METHODS FOR BIOENGINEERING\ecg_challenge\Data"
    analyze_excel_files(data_folder)
