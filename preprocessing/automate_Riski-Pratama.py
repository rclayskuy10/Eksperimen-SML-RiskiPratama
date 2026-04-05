"""
Automate Preprocessing - Iris Dataset
Nama: Riski Pratama
Script ini mengotomatisasi seluruh tahapan preprocessing yang dilakukan
pada notebook eksperimen.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import argparse


def load_data():
    """Load dataset Iris dari sklearn."""
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    print(f"[INFO] Dataset loaded: {df.shape}")
    return df, iris.feature_names


def remove_duplicates(df):
    """Menghapus baris duplikat."""
    before = len(df)
    df_clean = df.drop_duplicates()
    after = len(df_clean)
    print(f"[INFO] Duplikat dihapus: {before - after} baris (sisa: {after})")
    return df_clean


def detect_outliers(df, feature_names):
    """Deteksi outlier menggunakan metode IQR."""
    print("[INFO] Deteksi outlier (IQR method):")
    for col in feature_names:
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

    train_path = os.path.join(output_dir, 'iris_train_preprocessed.csv')
    test_path = os.path.join(output_dir, 'iris_test_preprocessed.csv')

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"[INFO] Saved: {train_path} ({train_df.shape})")
    print(f"[INFO] Saved: {test_path} ({test_df.shape})")
    return train_path, test_path


def main():
    parser = argparse.ArgumentParser(description='Automate Iris Preprocessing')
    parser.add_argument('--output-dir', type=str,
                        default='iris_preprocessing',
                        help='Direktori output untuk data preprocessing')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proporsi test set (default: 0.2)')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random state (default: 42)')
    args = parser.parse_args()

    print("=" * 60)
    print("AUTOMATE PREPROCESSING - IRIS DATASET")
    print("Nama: Riski Pratama")
    print("=" * 60)

    # 1. Load data
    df, feature_names = load_data()

    # 2. Hapus duplikat
    df_clean = remove_duplicates(df)

    # 3. Deteksi outlier
    df_clean = detect_outliers(df_clean, feature_names)

    # 4. Split data
    X_train, X_test, y_train, y_test = split_data(
        df_clean, feature_names,
        test_size=args.test_size,
        random_state=args.random_state
    )

    # 5. Standarisasi
    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_test, feature_names
    )

    # 6. Simpan hasil
    save_preprocessed(
        X_train_scaled, X_test_scaled, y_train, y_test,
        feature_names, args.output_dir
    )

    print("\n" + "=" * 60)
    print("PREPROCESSING SELESAI!")
    print("=" * 60)


if __name__ == "__main__":
    main()
