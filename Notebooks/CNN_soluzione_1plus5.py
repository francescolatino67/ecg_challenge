###############################################
# LIBRERIE
###############################################

import os
import numpy as np
import pandas as pd
import scipy
from scipy.signal import butter, filtfilt, iirnotch

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

from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", device)

###############################################
# PERCORSI AI DATI
###############################################

# Adatta questi percorsi alla tua struttura locale
ROOT_PATH   = os.path.dirname(os.getcwd())
DATA_FOLDER = os.path.join(ROOT_PATH, "Data")
DATA_Batch_01 = os.path.join(DATA_FOLDER, "01_batch_ECG_Signals")
DATA_Batch_02 = os.path.join(DATA_FOLDER, "02_batch_ECG_Signals")

###############################################
# FUNZIONI DI FILTRAGGIO ECG
###############################################

def butter_bandpass(lowcut, highcut, fs, order=4):
    """
    Crea un filtro passa-banda Butterworth.

    Parameters
    ----------
    lowcut : float
        Frequenza di taglio inferiore (Hz).
    highcut : float
        Frequenza di taglio superiore (Hz).
    fs : float
        Frequenza di campionamento (Hz).
    order : int
        Ordine del filtro.

    Returns
    -------
    b, a : ndarray
        Coefficienti del filtro IIR.
    """
    nyq = 0.5 * fs              # Frequenza di Nyquist
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, lowcut=1, highcut=40, fs=500, order=2):
    """
    Applica il filtro passa-banda ad un segnale monodimensionale.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data)

def notch_filter(data, freq=50, fs=500, quality_factor=30):
    """
    Filtro notch per rimuovere rumore di rete (es. 50 Hz).

    Parameters
    ----------
    freq : float
        Frequenza centrale da filtrare (es. 50 Hz).
    fs : float
        Frequenza di campionamento.
    quality_factor : float
        Fattore di qualità del filtro (più alto = banda più stretta).
    """
    b, a = iirnotch(freq / (fs / 2), quality_factor)
    return filtfilt(b, a, data)

###############################################
# CARICAMENTO E PRE-PROCESSING ECG
###############################################

def extract_patient_id(filename):
    """
    Estrae l'ID paziente dal nome del file .mat
    Assumiamo che il file sia tipo '123.mat' -> paziente 123
    """
    return int(filename.split(".")[0])

def load_and_preprocess_ecg():
    """
    Carica:
      - file Excel con dati tabellari (Batch1 + Batch2)
      - segnali ECG (.mat) per i due batch
    Applica:
      - rimozione media per derivazione
      - filtro passa-banda
      - filtro notch
    Concatena il tutto in:
      - signals: array (N_pazienti, T, 12)
      - tabular_data: DataFrame allineato a signals
    """

    # Caricamento tabellare
    filename_Batch_01 = os.path.join(DATA_FOLDER, "VALETUDO_database_1st_batch_en_all_info.xlsx")
    filename_Batch_02 = os.path.join(DATA_FOLDER, "VALETUDO_database_2nd_batch_en_all_info.xlsx")
    tabular_data_Batch_01 = pd.read_excel(filename_Batch_01)
    tabular_data_Batch_02 = pd.read_excel(filename_Batch_02)

    # Lista file .mat per ogni batch
    ECGs_1 = [f for f in os.listdir(DATA_Batch_01) if f.endswith(".mat")]
    ECGs_2 = [f for f in os.listdir(DATA_Batch_02) if f.endswith(".mat")]

    ECGs_1.sort(key=extract_patient_id)
    ECGs_2.sort(key=extract_patient_id)

    # Numero campioni e derivazioni; qui assumiamo 5000 x 12 come nel notebook
    n_samples = 5000
    n_leads   = 12

    signals_1 = np.empty((len(ECGs_1), n_samples, n_leads))
    signals_2 = np.empty((len(ECGs_2), n_samples, n_leads))

    # Parametri del segnale
    fs = 500  # Hz

    # Loop Batch 1
    for index, ecg_path in enumerate(ECGs_1):
        filepath = os.path.join(DATA_Batch_01, ecg_path)
        matdata  = scipy.io.loadmat(filepath)
        ecg      = matdata["val"]  # shape (T, 12)

        # Preprocessing per ciascuna derivazione
        for i in range(n_leads):
            # rimozione della media
            ecg[:, i] = ecg[:, i] - np.mean(ecg[:, i])
            # filtro passa-banda
            ecg[:, i] = apply_bandpass_filter(ecg[:, i], fs=fs)
            # filtro notch
            ecg[:, i] = notch_filter(ecg[:, i], fs=fs)

        signals_1[index, :, :] = ecg

    # Loop Batch 2
    for index, ecg_path in enumerate(ECGs_2):
        filepath = os.path.join(DATA_Batch_02, ecg_path)
        matdata  = scipy.io.loadmat(filepath)
        ecg      = matdata["val"]

        for i in range(n_leads):
            ecg[:, i] = ecg[:, i] - np.mean(ecg[:, i])
            ecg[:, i] = apply_bandpass_filter(ecg[:, i], fs=fs)
            ecg[:, i] = notch_filter(ecg[:, i], fs=fs)

        signals_2[index, :, :] = ecg

    # Concatenazione segnali e ordinamento dei dati tabellari
    signals = np.concatenate([signals_1, signals_2], axis=0)

    tabular_data = pd.concat(
        [
            tabular_data_Batch_01.sort_values(by="ECG_patient_id").reset_index(drop=True),
            tabular_data_Batch_02.sort_values(by="ECG_patient_id").reset_index(drop=True),
        ],
        ignore_index=True
    )

    print("Combined signals shape:", signals.shape)   # (N, 5000, 12)
    print("Combined tabular shape:", tabular_data.shape)

    return signals, tabular_data


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
        feature_cols = ["age_at_exam", "height", "weight", "trainning_load", "sex", "sport_classification"]
        X_train_final = X_train[feature_cols].copy()
        X_test_final  = X_test[feature_cols].copy()

        # Pulizia semplice su age_at_exam e trainning_load (come nel notebook)
        for df in [X_train_final, X_test_final]:
            df["age_at_exam"] = df["age_at_exam"].apply(lambda x: x if 0.0 <= x <= 100.0 else np.nan)
            df["trainning_load"] = df["trainning_load"].apply(lambda x: x if 0 < x <= 4 else np.nan)

        # Imputazione dei NaN (IterativeImputer)
        imputer = IterativeImputer(random_state=42)
        X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train_final), columns=feature_cols)
        X_test_imputed  = pd.DataFrame(imputer.transform(X_test_final), columns=feature_cols)

        # Definisco quali colonne sono numeriche e quali "categoriche"
        numeric_cols = ["age_at_exam", "height", "weight", "trainning_load"]
        categorical_cols = ["sex", "sport_classification"]

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


###############################################
# MAIN
###############################################

if __name__ == "__main__":
    # Carico e pre-processo i dati una sola volta
    signals, tabular_data = load_and_preprocess_ecg()

    # Verifico distribuzione classe
    n_pos = np.sum(tabular_data["sport_ability"] == 1)
    n_tot = len(tabular_data["sport_ability"])
    print(f"nb pos: {n_pos}")
    print(f"% pos: {n_pos / n_tot * 100:.2f}%")

    # Eseguo training + cross-validation
    train_and_evaluate(
        signals=signals,
        tabular_data=tabular_data,
        num_epochs=5,        # aumenta es. a 30–50 quando tutto funziona
        batch_size=32,
        n_splits=10,
        threshold=0.5        # puoi ottimizzarla in seguito
    )