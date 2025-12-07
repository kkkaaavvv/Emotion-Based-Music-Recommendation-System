# train_audio_models.py

import os
import json
import numpy as np
import librosa
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
import joblib

# -----------------------------
# CONFIG
# -----------------------------
AUDIO_DATASET_DIR = "audio_dataset"  # audio_dataset/emotion_name/*.wav
MODEL_DIR = "models/audio"

os.makedirs(MODEL_DIR, exist_ok=True)


# -----------------------------
# 1. Feature extraction
# -----------------------------
def extract_features(file_path, sr=22050, n_mfcc=20):
    """
    Load audio and compute MFCC-based feature vector (mean + std).
    """
    y, sr = librosa.load(file_path, sr=sr, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)  # (n_mfcc, T)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    feat = np.concatenate([mfcc_mean, mfcc_std])  # (2 * n_mfcc,)
    return feat


def load_dataset(root_dir):
    X = []
    y = []

    for emotion in os.listdir(root_dir):
        emotion_dir = os.path.join(root_dir, emotion)
        if not os.path.isdir(emotion_dir):
            continue

        for fname in os.listdir(emotion_dir):
            if not fname.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
                continue

            fpath = os.path.join(emotion_dir, fname)
            try:
                feat = extract_features(fpath)
                X.append(feat)
                y.append(emotion)
                print(f"[OK] {fpath} -> {emotion}")
            except Exception as e:
                print(f"[SKIP] {fpath} ({e})")

    X = np.array(X)
    y = np.array(y)
    return X, y


# -----------------------------
# 2. Train models
# -----------------------------
def main():
    print("Loading dataset...")
    X, y = load_dataset(AUDIO_DATASET_DIR)
    print("Dataset shape:", X.shape, y.shape)

    if len(X) == 0:
        raise RuntimeError(
            "No audio data found. Check AUDIO_DATASET_DIR path & that it has .wav/.mp3 files."
        )

    # Encode labels to integers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Save label encoder
    joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))
    print("Saved label encoder.")

    # ------- SVM (with scaling) -------
    print("Training SVM...")
    svm_clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, gamma="scale"))
    svm_clf.fit(X, y_encoded)
    joblib.dump(svm_clf, os.path.join(MODEL_DIR, "svm_model.pkl"))
    print("Saved SVM model.")

    # ------- Naive Bayes (Gaussian) -------
    print("Training Naive Bayes...")
    nb_clf = GaussianNB()
    nb_clf.fit(X, y_encoded)
    joblib.dump(nb_clf, os.path.join(MODEL_DIR, "naive_bayes_model.pkl"))
    print("Saved Naive Bayes model.")

    # ------- KMeans (unsupervised, then map clusters -> labels) -------
    print("Training KMeans...")
    n_classes = len(label_encoder.classes_)
    kmeans = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
    kmeans.fit(X)
    joblib.dump(kmeans, os.path.join(MODEL_DIR, "kmeans_model.pkl"))
    print("Saved KMeans model.")

    # Map each cluster index to the majority true label in that cluster
    cluster_labels = {}
    cluster_indices = kmeans.predict(X)
    for c in range(n_classes):
        indices = np.where(cluster_indices == c)[0]
        if len(indices) == 0:
            # fallback: assign most common overall label
            cluster_labels[c] = int(np.bincount(y_encoded).argmax())
        else:
            majority = int(np.bincount(y_encoded[indices]).argmax())
            cluster_labels[c] = majority

    with open(os.path.join(MODEL_DIR, "kmeans_cluster_map.json"), "w") as f:
        json.dump(cluster_labels, f)

    print("Saved KMeans cluster → label mapping.")
    print("All audio models trained and saved in:", MODEL_DIR)


if __name__ == "__main__":
    main()
