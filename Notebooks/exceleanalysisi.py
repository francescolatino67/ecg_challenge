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
    cols_to_drop = [
        'code', 'ECG_patient_id', 'AV block', 'ST abnormality', 
        'Complete BBB', 'Prolonged QTc', 'Supraventricular arrhythmias', 
        'Ventricular arrhythmias', 'Baseline ECG abnormalities'
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # One-Hot Encoding for 'training_load'
    if 'training_load' in df.columns:
        # Create dummies for 1, 2, 3, 4
        # Assuming training_load contains values like 1, 2, 3, 4
        dummies = pd.get_dummies(df['training_load'], prefix='training_load')
        # Ensure we have all 4 columns even if some values are missing in one batch
        for i in [1, 2, 3, 4]:
            col_name = f'training_load_{i}'
            if col_name not in dummies.columns:
                dummies[col_name] = 0
            # Convert to int (0/1) instead of bool
            dummies[col_name] = dummies[col_name].astype(int)
        
        # Select only the columns we want in the specific order
        dummies = dummies[['training_load_1', 'training_load_2', 'training_load_3']]
        
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

        # Sport Classification Analysis
        print(f"\n{'='*50}")
        print(f"Sport Classification Analysis (Combined Data)")
        print(f"{'='*50}")
        
        # Combine dataframes for this analysis to see overall patterns
        df_total = pd.concat([df1, df2], ignore_index=True)
        
        target_col = 'sport_ability'
        if target_col in df_total.columns:
            # Analyze against all other numeric columns
            numeric_cols = df_total.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                if col != target_col:
                    visualize_sport_classification(df_total, col, target_col)
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
