"""
Automate Preprocessing - Covertype Dataset
Nama: Riski Pratama
Script ini mengotomatisasi seluruh tahapan preprocessing yang dilakukan
pada notebook eksperimen. Dataset: Forest Covertype (sklearn).
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import os
import argparse


# 10 fitur kontinu pada Covertype
CONTINUOUS_FEATURES = [
    'Elevation', 'Aspect', 'Slope',
    'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
    'Horizontal_Distance_To_Roadways', 'Hillshade_9am',
    'Hillshade_Noon', 'Hillshade_3pm',
    'Horizontal_Distance_To_Fire_Points'
]


def load_data(n_samples=10000, random_state=42):
    """Load dataset Covertype dari sklearn dan subsample."""
    print("[INFO] Downloading/loading Covertype dataset...")
    covtype = fetch_covtype(as_frame=True)
    df_full = covtype.frame.copy()
    target_col = df_full.columns[-1]
    df_full = df_full.rename(columns={target_col: 'target'})
    print(f"[INFO] Full dataset: {df_full.shape}")

    # Subsample untuk efisiensi
    df = resample(df_full, n_samples=n_samples, random_state=random_state,
                  stratify=df_full['target'])
    df = df.reset_index(drop=True)
    print(f"[INFO] Subsampled dataset: {df.shape}")

    feature_names = [c for c in df.columns if c != 'target']
    return df, feature_names


def remove_duplicates(df):
    """Menghapus baris duplikat."""
    before = len(df)
    df_clean = df.drop_duplicates()
    after = len(df_clean)
    print(f"[INFO] Duplikat dihapus: {before - after} baris (sisa: {after})")
    return df_clean


def detect_outliers(df, feature_names):
    """Deteksi outlier menggunakan metode IQR (hanya fitur kontinu)."""
    print("[INFO] Deteksi outlier (IQR method) - fitur kontinu:")
    continuous = [f for f in feature_names if f in CONTINUOUS_FEATURES]
    for col in continuous:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        print(f"  - {col}: {len(outliers)} outlier")
    return df


def split_data(df, feature_names, test_size=0.2, random_state=42):
    """Split data menjadi train dan test set."""
    X = df[feature_names]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[INFO] Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test, feature_names):
    """Standarisasi fitur menggunakan StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("[INFO] Standarisasi selesai (StandardScaler)")
    return X_train_scaled, X_test_scaled, scaler


def save_preprocessed(X_train_scaled, X_test_scaled, y_train, y_test,
                      feature_names, output_dir):
    """Simpan data hasil preprocessing ke CSV."""
    os.makedirs(output_dir, exist_ok=True)

    train_df = pd.DataFrame(X_train_scaled, columns=feature_names)
    train_df['target'] = y_train.values

    test_df = pd.DataFrame(X_test_scaled, columns=feature_names)
    test_df['target'] = y_test.values

    train_path = os.path.join(output_dir, 'covtype_train_preprocessed.csv')
    test_path = os.path.join(output_dir, 'covtype_test_preprocessed.csv')

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"[INFO] Saved: {train_path} ({train_df.shape})")
    print(f"[INFO] Saved: {test_path} ({test_df.shape})")
    return train_path, test_path


def main():
    parser = argparse.ArgumentParser(description='Automate Covertype Preprocessing')
    parser.add_argument('--output-dir', type=str,
                        default='covtype_preprocessing',
                        help='Direktori output untuk data preprocessing')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proporsi test set (default: 0.2)')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random state (default: 42)')
    parser.add_argument('--n-samples', type=int, default=10000,
                        help='Jumlah sampel yang diambil (default: 10000)')
    args = parser.parse_args()

    print("=" * 60)
    print("AUTOMATE PREPROCESSING - COVERTYPE DATASET")
    print("Nama: Riski Pratama")
    print("=" * 60)

    df, feature_names = load_data(n_samples=args.n_samples,
                                  random_state=args.random_state)
    df_clean = remove_duplicates(df)
    df_clean = detect_outliers(df_clean, feature_names)

    X_train, X_test, y_train, y_test = split_data(
        df_clean, feature_names,
        test_size=args.test_size,
        random_state=args.random_state
    )

    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_test, feature_names
    )

    save_preprocessed(
        X_train_scaled, X_test_scaled, y_train, y_test,
        feature_names, args.output_dir
    )

    print("\n" + "=" * 60)
    print("PREPROCESSING SELESAI!")
    print("=" * 60)


if __name__ == "__main__":
    main()
