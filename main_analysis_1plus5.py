import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import os
import glob
import scipy

from tqdm import tqdm as tqdm
from scipy.signal import butter, filtfilt, iirnotch, find_peaks

from sklearn.model_selection import StratifiedKFold
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, recall_score, accuracy_score, confusion_matrix,
    balanced_accuracy_score, roc_auc_score, roc_curve
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", device)

# Set your own specific drive path
ROOT_PATH   = os.getcwd()
DATA_FOLDER   = os.path.join(ROOT_PATH, "Data")
DATA_Batch_01 = os.path.join(DATA_FOLDER, "01_batch_ECG_Signals")
DATA_Batch_02 = os.path.join(DATA_FOLDER, "02_batch_ECG_Signals")

# prepare functions for filtering
# basic ECG preprocessing

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, lowcut=1, highcut=40, fs=1000, order=2):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data, axis=0)

def notch_filter(data, freq=50, fs=1000, quality_factor=30):
    b, a = iirnotch(freq / (fs / 2), quality_factor)
    return filtfilt(b, a, data, axis=0)

# normalization function
def apply_normalization(data):
    """
    Applies Z-score normalization (Standardization) per lead.
    Formula: (x - mean) / std
    """
    # Calculate mean and std along the time axis (axis 0)
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    
    # Avoid division by zero by adding a small epsilon
    epsilon = 1e-8
    return (data - mu) / (sigma + epsilon)

def plot_12lead_for_patient(patient_id, signals, tabular_data, fs=500):
    """
    Plot 12-lead ECG for given patient_id using the already-merged `signals` array.
    Arrange leads so that limb leads (I, II, III, aVR, aVL, aVF) are in the first column
    and precordial leads (V1..V6) are in the second column.
    """
    # find index in tabular_data
    idx_list = tabular_data.index[tabular_data['ECG_patient_id'] == patient_id].tolist()
    if len(idx_list) == 0:
        raise ValueError(f"Patient id {patient_id} not found in tabular_data['ECG_patient_id'].")
    idx = int(idx_list[0])

    if idx < 0 or idx >= signals.shape[0]:
        raise IndexError(f"Found index {idx} out of range for signals (shape {signals.shape}).")

    sig = signals[idx]  # shape (n_samples, 12)
    if sig.ndim != 2 or sig.shape[1] != 12:
        raise ValueError(f"Expected signal shape (n_samples, 12), got {sig.shape}")

    n_samples = sig.shape[0]
    t = np.arange(n_samples) / fs  # seconds; change to ms if desired by multiplying by 1000

    # desired layout:
    # column 0: I, II, III, aVR, aVL, aVF
    # column 1: V1, V2, V3, V4, V5, V6
    lead_names = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']

    # mapping lead index -> (row, col) in 6x2 grid
    mapping = {
        0: (0, 0),  # I
        1: (1, 0),  # II
        2: (2, 0),  # III
        3: (3, 0),  # aVR
        4: (4, 0),  # aVL
        5: (5, 0),  # aVF
        6: (0, 1),  # V1
        7: (1, 1),  # V2
        8: (2, 1),  # V3
        9: (3, 1),  # V4
        10: (4, 1), # V5
        11: (5, 1)  # V6
    }

    # vertical offsets for visualization
    amp = np.max(np.abs(sig)) if np.max(np.abs(sig)) > 0 else 1.0
    offset = 2 * amp
    print(offset)

    fig, axes = plt.subplots(6, 2, figsize=(12, 10), sharex=True)
    for lead_idx in range(12):
        r, c = mapping[lead_idx]
        ax = axes[r, c]
        ax.plot(t, sig[:, lead_idx], color='#155874')
        ax.set_ylabel(lead_names[lead_idx], rotation=0, labelpad=16, va='center')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    ax = axes[0, 0]
    sig_plot = sig[:, 6] + 6 * offset
    sig_plot = sig_plot - sig_plot.mean(axis=0, keepdims=True)
    ax.plot(t, sig_plot, color='red')

    # remove empty spines/labels for clarity
    for r in range(6):
        for c in range(2):
            axes[r, c].tick_params(axis='y', which='both', left=True)
    axes[-1, 0].set_xlabel('Time (s)')
    axes[-1, 1].set_xlabel('Time (s)')

    fig.suptitle(f'12-lead ECG Patient ID {patient_id} (index {idx})', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()

# Tabular data management functions
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

def preprocess_ecg_batch(signals, fs=1000):
    """
    Args:
        signals: np.array of shape (N_subjects, 5000, 12)
        fs: Sampling frequency (default 1000 Hz)
    Returns:
        processed_signals: np.array of same shape
    """
    processed_signals = np.empty_like(signals)
    
    print(f"Starting preprocessing for {len(signals)} subjects...")
    
    for i, patient_ecg in tqdm(enumerate(signals)):
        # patient_ecg shape is (5000, 12)
        
        # 1. Bandpass Filter (0.5-40 Hz as per your config)
        # Note: 1Hz lowcut is aggressive and may attenuate ST changes. 
        # 0.5Hz is safer for clinical ischemia detection, but 1Hz is cleaner for noise.
        filtered = apply_bandpass_filter(patient_ecg, lowcut=0.5, highcut=40, fs=fs)
        
        # 2. Notch Filter (50 Hz)
        filtered = notch_filter(filtered, freq=50, fs=fs)
        
        # 3. Normalization (Z-score)
        normalized = apply_normalization(filtered)
        
        processed_signals[i] = normalized
        
    print("Preprocessing complete.")
    return processed_signals

#import the data and filter the signals
# This could be changed depending if you download or not the data

filename_Batch_01     = f"{DATA_FOLDER}\\VALETUDO_database_1st_batch_en_all_info.xlsx"
filename_Batch_02     = f"{DATA_FOLDER}\\VALETUDO_database_2nd_batch_en_all_info.xlsx"
tabular_data_Batch_01 = pd.read_excel(filename_Batch_01)
tabular_data_Batch_02 = pd.read_excel(filename_Batch_02)

# Concatenate and balance the dataset (keeping only the balanced data for further analysis)
tabular_data = pd.concat([
    tabular_data_Batch_01.sort_values(by="ECG_patient_id").reset_index(drop=True),
    tabular_data_Batch_02.sort_values(by="ECG_patient_id").reset_index(drop=True)
], ignore_index=True)
tabular_data_balanced, removed_ids = balance_dataset(tabular_data, target_col='sport_ability', id_col='ECG_patient_id', random_state=42)

tabular_data_balanced = preprocess_data(tabular_data_balanced)

# --- Load and filter both batches ---
ECGs_1 = [f for f in os.listdir(DATA_Batch_01) if f.endswith(".mat") and int(f.split(".")[0]) not in removed_ids]
ECGs_2 = [f for f in os.listdir(DATA_Batch_02) if f.endswith(".mat") and int(f.split(".")[0]) not in removed_ids]

def extract_patient_id(filename):
    return int(filename.split(".")[0])

ECGs_1.sort(key=extract_patient_id)
ECGs_2.sort(key=extract_patient_id)

signals_1 = np.empty((len(ECGs_1), 5000, 12))
signals_2 = np.empty((len(ECGs_2), 5000, 12))

for index, ecg_path in enumerate(ECGs_1):
    filepath = os.path.join(DATA_Batch_01, ecg_path)
    matdata = scipy.io.loadmat(filepath)
    ecg = matdata['val']
    signals_1[index, :, :] = ecg

for index, ecg_path in enumerate(ECGs_2):
    filepath = os.path.join(DATA_Batch_02, ecg_path)
    matdata = scipy.io.loadmat(filepath)
    ecg = matdata['val']
    signals_2[index, :, :] = ecg

# --- Concatenate signals and tabular data ---
signals = np.concatenate([signals_1, signals_2], axis=0)

print("Combined tabular shape:", tabular_data.shape)
print("Combined signals shape:", signals.shape)
print("Balanced tabular shape:", tabular_data_balanced.shape)

print(f"nb pos: {np.sum(tabular_data_Batch_01['sport_ability']==1)}")
print(f"% pos: {np.sum(tabular_data_Batch_01['sport_ability']==1)/len(tabular_data_Batch_01['sport_ability'])*100:.2f}%")

# Example: plot patient with id 4
# plot_12lead_for_patient(4, signals, tabular_data_Batch_01, fs=1000)

signals_clean = preprocess_ecg_batch(signals, fs=1000)

# plot_12lead_for_patient(4, signals_clean, tabular_data_Batch_01, fs=1000)

def plot_comparison_for_patient(patient_id, signals_raw, signals_clean, tabular_data, fs=1000):
    """
    Plot comparison of Raw vs Cleaned 12-lead ECG for given patient_id WITHOUT temporary normalization.
    Arrange leads so that limb leads (I, II, III, aVR, aVL, aVF) are in the first column
    and precordial leads (V1..V6) are in the second column.
    """
    # find index in tabular_data
    idx_list = tabular_data.index[tabular_data['ECG_patient_id'] == patient_id].tolist()
    if len(idx_list) == 0:
        raise ValueError(f"Patient id {patient_id} not found in tabular_data['ECG_patient_id'].")
    idx = int(idx_list[0])

    if idx < 0 or idx >= signals_raw.shape[0]:
        raise IndexError(f"Found index {idx} out of range for signals (shape {signals_raw.shape}).")

    sig_raw = signals_raw[idx]    # shape (n_samples, 12)
    sig_clean = signals_clean[idx] # shape (n_samples, 12)

    if sig_raw.ndim != 2 or sig_raw.shape[1] != 12:
        raise ValueError(f"Expected signal shape (n_samples, 12), got {sig_raw.shape}")

    n_samples = sig_raw.shape[0]
    t = np.arange(n_samples) / fs  # seconds

    # desired layout:
    # column 0: I, II, III, aVR, aVL, aVF
    # column 1: V1, V2, V3, V4, V5, V6
    lead_names = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']

    # mapping lead index -> (row, col) in 6x2 grid
    mapping = {
        0: (0, 0),  # I
        1: (1, 0),  # II
        2: (2, 0),  # III
        3: (3, 0),  # aVR
        4: (4, 0),  # aVL
        5: (5, 0),  # aVF
        6: (0, 1),  # V1
        7: (1, 1),  # V2
        8: (2, 1),  # V3
        9: (3, 1),  # V4
        10: (4, 1), # V5
        11: (5, 1)  # V6
    }

    fig, axes = plt.subplots(6, 2, figsize=(12, 10), sharex=True)
    
    for lead_idx in range(12):
        r, c = mapping[lead_idx]
        ax = axes[r, c]
        
        # We assume signals_clean is already Z-scored. We apply Z-score to raw just for the plot.
        raw_trace = sig_raw[:, lead_idx]
        raw_vis = (raw_trace - np.mean(raw_trace)) / (np.std(raw_trace) + 1e-8)

        ax.plot(t, raw_vis, color='lightgrey', label='Raw', linewidth=1.5, alpha=0.8)
        
        # Plot Cleaned (Foreground, Your Style Color)
        ax.plot(t, sig_clean[:, lead_idx], color='#155874', label='Cleaned', linewidth=1.0)
        
        ax.set_ylabel(lead_names[lead_idx], rotation=0, labelpad=16, va='center')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add legend only to the first subplot to avoid clutter
        if lead_idx == 0:
            ax.legend(loc='upper right', frameon=False, fontsize=8)

    # remove empty spines/labels for clarity
    for r in range(6):
        for c in range(2):
            axes[r, c].tick_params(axis='y', which='both', left=True)
            
    axes[-1, 0].set_xlabel('Time (s)')
    axes[-1, 1].set_xlabel('Time (s)')

    fig.suptitle(f'Raw vs Cleaned ECG - Patient ID {patient_id}', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()

# --- Usage Example ---
# Assuming 'signals' is your raw data and 'signals_clean' is the output of preprocess_ecg_batch
# plot_comparison_for_patient(4, signals, signals_clean, tabular_data_Batch_01, fs=1000)

def _get_patient_signal(patient_id, signals, tabular_data):
    """Internal helper to safely get patient index and signal."""
    idx_list = tabular_data.index[tabular_data['ECG_patient_id'] == patient_id].tolist()
    if len(idx_list) == 0:
        raise ValueError(f"Patient id {patient_id} not found.")
    idx = int(idx_list[0])
    
    if idx < 0 or idx >= signals.shape[0]:
        raise IndexError(f"Index {idx} out of range.")
        
    return signals[idx]

def plot_raw_leads_separate(patient_id, signals_raw, tabular_data, fs=1000):
    """
    Generates 12 separate figures for the RAW ECG leads.
    No normalization is applied.
    """
    sig = _get_patient_signal(patient_id, signals_raw, tabular_data)
    t = np.arange(sig.shape[0]) / fs
    lead_names = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']

    print(f"--- RAW ECG Signals for Patient {patient_id} ---")
    
    for i, name in enumerate(lead_names):
        plt.figure(figsize=(10, 3)) # Wider, shorter aspect ratio for single lead

        ax = plt.gca()
        # Plot raw signal directly (no z-score)
        ax.plot(t if i < 6 else t + 5, sig[:, i], color='black', linewidth=1)
        
        # Turn off top, right, and bottom spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        
        # Keep only the left spine
        ax.spines['left'].set_visible(True)
        ax.spines['left'].set_color('#333333')   # Dark grey instead of harsh black

        ax.tick_params(axis='both', labelsize=12, length=0)  # Remove tick marks for cleaner look

        ax.set_title(f"Lead {name} (Raw)", loc='left', fontweight='bold', fontsize=16)
        if i < 6:
            ax.set_xlim(0,5)  # Show only the first 5 seconds for better visibility
        else:
            ax.set_xlim(5,10)  # Show only the first 5 seconds for better visibility
        ax.set_xlabel("Time [s]", size=16)
        ax.set_ylabel("Amplitude [mV]", size=16)
        ax.grid(True, which='both', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()

def plot_clean_leads_separate(patient_id, signals_clean, tabular_data, fs=1000):
    """
    Generates 12 separate figures for the CLEANED ECG leads.
    """
    sig = _get_patient_signal(patient_id, signals_clean, tabular_data)
    t = np.arange(sig.shape[0]) / fs
    lead_names = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']

    print(f"--- CLEANED ECG Signals for Patient {patient_id} ---")

    for i, name in enumerate(lead_names):
        plt.figure(figsize=(10, 3))

        ax = plt.gca()
        
        # Plot cleaned signal in your preferred blue
        ax.plot(t if i < 6 else t + 5, sig[:, i], color='#155874', linewidth=1.2)
        
        # Turn off top, right, and bottom spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        
        # Keep only the left spine
        ax.spines['left'].set_visible(True)
        ax.spines['left'].set_color('#333333')   # Dark grey instead of harsh black

        ax.tick_params(axis='both', labelsize=12, length=0)  # Remove tick marks for cleaner look
        
        ax.set_title(f"Lead {name} (Cleaned)", loc='left', fontweight='bold', color='#155874', fontsize=16)
        if i < 6:
            ax.set_xlim(0,5)  # Show only the first 5 seconds for better visibility
        else:
            ax.set_xlim(5,10)  # Show only the first 5 seconds for better visibility
        ax.set_xlabel("Time [s]", size=16)
        ax.set_ylabel("Normalized Amplitude [-]", size=16)
        ax.grid(True, which='both', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()

# --- Usage ---
# plot_raw_leads_separate(4, signals, tabular_data_Batch_01)
# plot_clean_leads_separate(4, signals_clean, tabular_data_Batch_01)

###############################################
# SEGMENTAZIONE DEL SEGNALE
###############################################

def segment_ecg(signals, segment_length=2500, start_mode="begin"):
    """
    Segmenta i segnali ECG lungo l'asse temporale.

    Parameters
    ----------
    signals : np.ndarray
        Shape (N_pazienti, T, 12)
    segment_length : int
        Lunghezza del segmento (in campioni).
    start_mode : str
        Modalità per scegliere lo start del segmento per ogni paziente.
        "begin" -> si parte da 0 (come nel notebook)
        In futuro puoi usare "middle", "random" ecc.

    Returns
    -------
    segments : np.ndarray
        Shape (N_pazienti, segment_length, 12)
    """
    N, T, C = signals.shape
    segments = np.zeros((N, segment_length, C))

    for i in range(N):
        if start_mode == "begin":
            start = 0
        elif start_mode == "middle":
            start = max(0, (T - segment_length) // 2)
        else:
            # di default: begin
            start = 0

        end = start + segment_length
        if end > T:
            # se segment_length è più lungo del segnale, facciamo un semplice crop finale
            end = T
            start = T - segment_length

        segments[i, :, :] = signals[i, start:end, :]

    return segments

###############################################
# DATASET PYTORCH
###############################################

class ECGDataset(Dataset):
    """
    Dataset PyTorch per combinare:
      - feature tabellari
      - segmenti ECG
      - etichette (sport_ability)
    """

    def __init__(self, tabular_df, signals, labels):
        """
        Parameters
        ----------
        tabular_df : pd.DataFrame
            DataFrame con le feature tabellari selezionate.
        signals : np.ndarray
            Array numpy con i segmenti ECG (N, T, 12).
        labels : pd.Series
            Serie con le etichette binarie (0/1).
        """
        # signals: (N, T, 12) -> PyTorch vuole (N, C, T)
        self.signals = torch.tensor(signals, dtype=torch.float32).permute(0, 2, 1)
        self.labels  = torch.tensor(labels.values, dtype=torch.float32).unsqueeze(1)
        self.tabular = torch.tensor(tabular_df.values, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        tab = self.tabular[idx]
        sig = self.signals[idx]
        lab = self.labels[idx]
        return tab, sig, lab


###############################################
# ARCHITETTURA DEL MODELLO
###############################################

class ECGBackbone(nn.Module):
    """
    Backbone per il segnale ECG:
      - CNN 1D su 12 derivazioni
      - GRU bidirezionale per catturare la dinamica temporale
    Output: embedding vettoriale per ogni paziente
    """

    def __init__(self, in_channels=12, hidden_size=128):
        super().__init__()

        # Blocco di convoluzioni 1D + pooling
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(2),  # T -> T/2

            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),  # T -> T/4

            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(2),  # T -> T/8
        )

        # GRU bidirezionale sul tempo
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (B, 12, T_segment)

        Returns
        -------
        h_flat : torch.Tensor
            Shape (B, 2*hidden_size)
        """
        # CNN 1D
        x = self.conv_block(x)   # (B, 128, T')
        # Permuta a (B, T', C) per la GRU
        x = x.permute(0, 2, 1)   # (B, T', 128)

        # GRU
        # output: (B, T', 2*hidden_size)
        # h: (num_layers*2, B, hidden_size) -> ultima hidden state per direzione
        _, h = self.gru(x)

        # h ha shape (2, B, hidden_size) dato num_layers=1, bidirectional=True
        h = h.permute(1, 0, 2).reshape(x.size(0), -1)  # (B, 2*hidden_size)
        return h


class TabularBranch(nn.Module):
    """
    Ramo MLP per le feature tabellari.
    """

    def __init__(self, in_features, hidden_dim=32):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.GELU(),
            nn.Linear(32, hidden_dim),
            nn.GELU()
        )

    def forward(self, x):
        """
        x: (B, in_features)
        return: (B, hidden_dim)
        """
        return self.net(x)


class ECGSportAbilityModel(nn.Module):
    """
    Modello multimodale:
      - ECGBackbone: embedding ECG (2*ecg_hidden)
      - TabularBranch: embedding tabellare (tab_hidden)
      - Classifier: MLP finale che concatena le due embedding
    Output: probabilità di sport_ability = 1
    """

    def __init__(self, tab_in_features, ecg_hidden=128, tab_hidden=32, dropout=0.3):
        super().__init__()

        self.ecg_backbone = ECGBackbone(in_channels=12, hidden_size=ecg_hidden)
        self.tab_branch   = TabularBranch(in_features=tab_in_features, hidden_dim=tab_hidden)

        fusion_dim = 2 * ecg_hidden + tab_hidden

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, tab, ecg):
        """
        Parameters
        ----------
        tab : torch.Tensor
            Feature tabellari, shape (B, F_tab)
        ecg : torch.Tensor
            Segmenti ECG, shape (B, 12, T)

        Returns
        -------
        prob : torch.Tensor
            Probabilità di classe positiva (B, 1)
        """
        ecg_emb = self.ecg_backbone(ecg)      # (B, 2*ecg_hidden)
        tab_emb = self.tab_branch(tab)        # (B, tab_hidden)

        x = torch.cat([ecg_emb, tab_emb], dim=1)  # (B, fusion_dim)
        logits = self.classifier(x)               # (B, 1)
        prob = torch.sigmoid(logits)              # Sigmoid per avere probabilità
        return prob


###############################################
# TRAINING + CROSS-VALIDATION
###############################################

def train_and_evaluate(signals, tabular_data, num_epochs=5, batch_size=32, n_splits=10, threshold=0.5):
    """
    Esegue cross-validation stratificata sul target 'sport_ability'
    usando il modello ECGSportAbilityModel.

    Parameters
    ----------
    signals : np.ndarray
        Array (N, T, 12) con tutti i segnali filtrati.
    tabular_data : pd.DataFrame
        DataFrame con le colonne cliniche e il target 'sport_ability'.
    num_epochs : int
        Numero di epoche per ogni fold.
    batch_size : int
        Dimensione del batch.
    n_splits : int
        Numero di fold per StratifiedKFold.
    threshold : float
        Soglia per binarizzare le probabilità (per le metriche).
    """

    # Lista dove accumulare metriche per ogni fold
    f1_list_all_folds = []
    f1_list_all_folds_train = []
    sensitivity_list_all_folds = []
    sensitivity_list_all_folds_train = []
    specificity_list_all_folds = []
    specificity_list_all_folds_train = []
    accuracy_list_all_folds = []
    accuracy_list_all_folds_train = []
    auc_score_list_all_folds = []
    auc_score_list_all_folds_train = []
    fpr_list_all_folds = []
    tpr_list_all_folds = []
    test_loss_all_folds = []
    train_loss_all_folds = []
    train_loss_max = []
    test_loss_max = []
    epochs_all_fold = []

    # Stratified K-Fold su sport_ability
    strat_kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Per ogni fold:
    for fold_idx, (train_index, test_index) in enumerate(
        strat_kf.split(tabular_data, tabular_data["sport_ability"])
    ):
        print(f"\n====================== Fold {fold_idx + 1}/{n_splits} ======================\n")

        # Split tabellare
        X_train = tabular_data.iloc[train_index, :].copy()
        X_test  = tabular_data.iloc[test_index, :].copy()

        # Split segnali
        ecg_train = signals[train_index, :, :]  # (N_train, T, 12)
        ecg_test  = signals[test_index, :, :]   # (N_test,  T, 12)

        # Segmentazione ECG (puoi cambiare segment_length)
        segment_length = 2500
        ecg_train_segments = segment_ecg(ecg_train, segment_length=segment_length, start_mode="begin")
        ecg_test_segments  = segment_ecg(ecg_test,  segment_length=segment_length, start_mode="begin")

        # Target
        Y_train = X_train["sport_ability"]
        Y_test  = X_test["sport_ability"]

        # Seleziona le colonne che vuoi usare come feature tabellari
        # In questo esempio uso: age_at_exam, height, weight, trainning_load, sex, sport_classification
        feature_cols = tabular_data.columns.difference(["ECG_patient_id", "sport_ability"])
        print(feature_cols)
        X_train_final = X_train[feature_cols].copy()
        X_test_final  = X_test[feature_cols].copy()

        # Pulizia semplice su age_at_exam e trainning_load (come nel notebook)
        for df in [X_train_final, X_test_final]:
            df["age_at_exam"] = df["age_at_exam"].apply(lambda x: x if 0.0 <= x <= 100.0 else np.nan)
            # df["trainning_load"] = df["trainning_load"].apply(lambda x: x if 0 < x <= 4 else np.nan)

        # Imputazione dei NaN (IterativeImputer)
        imputer = IterativeImputer(random_state=42)
        X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train_final), columns=feature_cols)
        X_test_imputed  = pd.DataFrame(imputer.transform(X_test_final), columns=feature_cols)

        # Definisco quali colonne sono numeriche e quali "categoriche"
        numeric_cols = ["age_at_exam", "height", "weight"]
        categorical_cols = [col for col in feature_cols if col not in numeric_cols]        

        # Scaling sulle numeriche
        scaler = StandardScaler()
        X_train_imputed[numeric_cols] = scaler.fit_transform(X_train_imputed[numeric_cols])
        X_test_imputed[numeric_cols]  = scaler.transform(X_test_imputed[numeric_cols])

        # Piccola trasformazione per le categoriche se necessario
        # (ad esempio mappare 0 a -1 se vogliamo distinguerlo da "missing")
        for col in categorical_cols:
            X_train_imputed[col] = X_train_imputed[col].apply(lambda x: -1 if x == 0 else x)
            X_test_imputed[col]  = X_test_imputed[col].apply(lambda x: -1 if x == 0 else x)

        # DataFrame finale per PyTorch (concateno numeric + categorical nell'ordine che preferisco)
        train_final_df = pd.concat(
            [X_train_imputed[numeric_cols], X_train_imputed[categorical_cols]], axis=1
        )
        test_final_df = pd.concat(
            [X_test_imputed[numeric_cols], X_test_imputed[categorical_cols]], axis=1
        )

        # Creo Dataset e DataLoader
        train_dataset = ECGDataset(train_final_df, ecg_train_segments, Y_train)
        test_dataset  = ECGDataset(test_final_df,  ecg_test_segments,  Y_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

        # Inizializzo il modello per questo fold
        tab_in_features = train_final_df.shape[1]  # numero di colonne tabellari
        model = ECGSportAbilityModel(tab_in_features=tab_in_features).to(device)

        # Uso BCE con probabilità (il modello ha già la sigmoid)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Per salvare metriche per epoca (dentro il singolo fold)
        f1_list_single_fold = []
        f1_list_single_fold_train = []
        sensitivity_list_single_fold = []
        sensitivity_list_single_fold_train = []
        specificity_list_single_fold = []
        specificity_list_single_fold_train = []
        accuracy_list_single_fold = []
        accuracy_list_single_fold_train = []
        auc_score_list_single_fold = []
        auc_score_list_single_fold_train = []
        fpr_list_single_fold = []
        tpr_list_single_fold = []
        train_loss_single_fold = []
        test_loss_single_fold = []
        epochs_single_fold = []

        # Loop di training per epoche
        for epoch in tqdm(range(num_epochs), desc=f"Fold {fold_idx+1}/{n_splits}"):
            ###################################
            # FASE DI TRAINING
            ###################################
            model.train()
            train_loss = 0.0

            all_labels_train = []
            all_preds_train = []
            all_outputs_train = []

            for tab_batch, sig_batch, lab_batch in train_loader:
                tab_batch = tab_batch.to(device)
                sig_batch = sig_batch.to(device)
                lab_batch = lab_batch.to(device)

                optimizer.zero_grad()
                outputs = model(tab_batch, sig_batch)  # (B,1) probabilità

                loss = criterion(outputs, lab_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                # Conversione per metriche
                probs = outputs.detach().cpu().numpy().ravel()
                preds = (probs > threshold).astype(int)
                labs  = lab_batch.detach().cpu().numpy().ravel().astype(int)

                all_outputs_train.extend(probs)
                all_preds_train.extend(preds)
                all_labels_train.extend(labs)

            avg_train_loss = train_loss / len(train_loader)
            train_loss_single_fold.append(avg_train_loss)

            # Metriche train
            train_accuracy = accuracy_score(all_labels_train, all_preds_train) * 100
            f1_train = f1_score(all_labels_train, all_preds_train)
            tn, fp, fn, tp = confusion_matrix(all_labels_train, all_preds_train).ravel()
            sensitivity_train = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity_train = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            auc_score_train = roc_auc_score(all_labels_train, all_outputs_train)

            f1_list_single_fold_train.append(f1_train)
            sensitivity_list_single_fold_train.append(sensitivity_train)
            specificity_list_single_fold_train.append(specificity_train)
            accuracy_list_single_fold_train.append(train_accuracy)
            auc_score_list_single_fold_train.append(auc_score_train)

            print(f"\n[Fold {fold_idx+1}] Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}")
            print(
                f"Train Acc: {train_accuracy:.2f}% | F1: {f1_train:.4f} | "
                f"Sens: {sensitivity_train:.4f} | Spec: {specificity_train:.4f} | AUC: {auc_score_train:.4f}"
            )

            ###################################
            # FASE DI VALIDAZIONE
            ###################################
            model.eval()
            test_loss = 0.0

            all_labels_test = []
            all_preds_test = []
            all_outputs_test = []

            with torch.no_grad():
                for tab_batch, sig_batch, lab_batch in test_loader:
                    tab_batch = tab_batch.to(device)
                    sig_batch = sig_batch.to(device)
                    lab_batch = lab_batch.to(device)

                    outputs = model(tab_batch, sig_batch)
                    loss = criterion(outputs, lab_batch)
                    test_loss += loss.item()

                    probs = outputs.detach().cpu().numpy().ravel()
                    preds = (probs > threshold).astype(int)
                    labs  = lab_batch.detach().cpu().numpy().ravel().astype(int)

                    all_outputs_test.extend(probs)
                    all_preds_test.extend(preds)
                    all_labels_test.extend(labs)

            avg_test_loss = test_loss / len(test_loader)
            test_loss_single_fold.append(avg_test_loss)

            # Metriche test
            test_accuracy = accuracy_score(all_labels_test, all_preds_test) * 100
            f1 = f1_score(all_labels_test, all_preds_test)
            tn, fp, fn, tp = confusion_matrix(all_labels_test, all_preds_test).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            auc_score = roc_auc_score(all_labels_test, all_outputs_test)
            fpr, tpr, _ = roc_curve(all_labels_test, all_outputs_test)

            f1_list_single_fold.append(f1)
            sensitivity_list_single_fold.append(sensitivity)
            specificity_list_single_fold.append(specificity)
            accuracy_list_single_fold.append(test_accuracy)
            auc_score_list_single_fold.append(auc_score)
            fpr_list_single_fold.append(fpr)
            tpr_list_single_fold.append(tpr)
            epochs_single_fold.append(epoch)

            print(
                f"Test Loss: {avg_test_loss:.4f} | Acc: {test_accuracy:.2f}% | "
                f"F1: {f1:.4f} | Sens: {sensitivity:.4f} | Spec: {specificity:.4f} | AUC: {auc_score:.4f}"
            )
            print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")

        # Seleziono l'epoca con F1 massima sul test per questo fold
        max_f1 = max(f1_list_single_fold)
        max_f1_index = f1_list_single_fold.index(max_f1)

        f1_list_all_folds.append(f1_list_single_fold[max_f1_index])
        f1_list_all_folds_train.append(f1_list_single_fold_train[max_f1_index])
        sensitivity_list_all_folds.append(sensitivity_list_single_fold[max_f1_index])
        sensitivity_list_all_folds_train.append(sensitivity_list_single_fold_train[max_f1_index])
        specificity_list_all_folds.append(specificity_list_single_fold[max_f1_index])
        specificity_list_all_folds_train.append(specificity_list_single_fold_train[max_f1_index])
        accuracy_list_all_folds.append(accuracy_list_single_fold[max_f1_index])
        accuracy_list_all_folds_train.append(accuracy_list_single_fold_train[max_f1_index])
        auc_score_list_all_folds.append(auc_score_list_single_fold[max_f1_index])
        auc_score_list_all_folds_train.append(auc_score_list_single_fold_train[max_f1_index])
        fpr_list_all_folds.append(fpr_list_single_fold[max_f1_index])
        tpr_list_all_folds.append(tpr_list_single_fold[max_f1_index])
        test_loss_all_folds.append(test_loss_single_fold)
        test_loss_max.append(test_loss_single_fold[max_f1_index])
        train_loss_all_folds.append(train_loss_single_fold)
        train_loss_max.append(train_loss_single_fold[max_f1_index])
        epochs_all_fold.append(epochs_single_fold[max_f1_index])

    ###################################
    # REPORT FINALE SU TUTTI I FOLD
    ###################################
    print("\n====================== RISULTATI FINALI (per fold, epoca di max F1) ======================\n")
    print(f"Accuracy Test:  {accuracy_list_all_folds}")
    print(f"Accuracy Train: {accuracy_list_all_folds_train}")
    print(f"F1 Test:        {f1_list_all_folds}")
    print(f"F1 Train:       {f1_list_all_folds_train}")
    print(f"Sens Test:      {sensitivity_list_all_folds}")
    print(f"Sens Train:     {sensitivity_list_all_folds_train}")
    print(f"Spec Test:      {specificity_list_all_folds}")
    print(f"Spec Train:     {specificity_list_all_folds_train}")
    print(f"AUC Test:       {auc_score_list_all_folds}")
    print(f"AUC Train:      {auc_score_list_all_folds_train}")
    print(f"Test Loss (epoca max F1):  {test_loss_max}")
    print(f"Train Loss (epoca max F1): {train_loss_max}")
    print(f"Epoche selezionate per fold: {epochs_all_fold}")

    ###################################
    # PLOT ROC PER TUTTI I FOLD
    ###################################
    plt.figure(figsize=(10, 8))

    colors = [
        "#E32947", "#F4A9B5", "#155874", "#29ABE2", "#E38D29",
        "#E3DA29", "#7FE329", "#BFBFBF", "#00B050", "#7030A0", "#996633"
    ]
    while len(colors) < len(fpr_list_all_folds):
        colors += colors

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []

    for i in range(len(fpr_list_all_folds)):
        interp_tpr = np.interp(mean_fpr, fpr_list_all_folds[i], tpr_list_all_folds[i])
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        auc_val = auc_score_list_all_folds[i]
        aucs.append(auc_val)
        plt.plot(
            fpr_list_all_folds[i],
            tpr_list_all_folds[i],
            color=colors[i],
            label=f"Fold {i+1} (AUC={auc_val:.2f})"
        )

    plt.plot([0, 1], [0, 1], color="black", linestyle="--", label="Chance")

    tprs = np.array(tprs)
    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    plt.plot(
        mean_fpr,
        mean_tpr,
        color="#E32947",
        label=f"Mean ROC (AUC={mean_auc:.2f}±{std_auc:.2f})",
        linewidth=2
    )
    plt.fill_between(
        mean_fpr,
        mean_tpr - std_tpr,
        mean_tpr + std_tpr,
        color="#E32947",
        alpha=0.2,
        label="±1 std"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve per tutti i fold")
    plt.legend(loc="lower right")
    plt.show()

    ###################################
    # PLOT LEARNING CURVES (LOSS)
    ###################################
    plt.figure(figsize=(10, 6))
    for i in range(len(train_loss_all_folds)):
        plt.plot(train_loss_all_folds[i], color=colors[i % len(colors)], alpha=0.8, label=f"Train Fold {i+1}")
        plt.plot(test_loss_all_folds[i], color=colors[i % len(colors)], alpha=0.8, linestyle="--", label=f"Test Fold {i+1}")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curves (Train/Test Loss per Fold)")
    plt.legend(ncol=2, fontsize=8)
    plt.grid(True)
    plt.show()

# Verifico distribuzione classe
n_pos = np.sum(tabular_data_balanced["sport_ability"] == 1)
n_tot = len(tabular_data_balanced["sport_ability"])
n_tot_signals = len(signals)
print(f"nb pos: {n_pos}")
print(f"% pos: {n_pos / n_tot * 100:.2f}%")

print(f"Total samples in signals: {n_tot_signals}")

# Eseguo training + cross-validation
train_and_evaluate(
    signals=signals,
    tabular_data=tabular_data_balanced,
    num_epochs=5,        # aumenta es. a 30–50 quando tutto funziona
    batch_size=32,
    n_splits=10,
    threshold=0.5        # puoi ottimizzarla in seguito
)