# predict_audio.py

import os
import json
from typing import Tuple

import numpy as np
import librosa
import joblib

MODEL_DIR = "models/audio"

SVM_PATH = os.path.join(MODEL_DIR, "svm_model.pkl")
NB_PATH = os.path.join(MODEL_DIR, "naive_bayes_model.pkl")
KMEANS_PATH = os.path.join(MODEL_DIR, "kmeans_model.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
KMEANS_MAP_PATH = os.path.join(MODEL_DIR, "kmeans_cluster_map.json")

# Load models once at import
svm_model = joblib.load(SVM_PATH)
nb_model = joblib.load(NB_PATH)
kmeans_model = joblib.load(KMEANS_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

with open(KMEANS_MAP_PATH, "r") as f:
    kmeans_cluster_map = {int(k): int(v) for k, v in json.load(f).items()}


def extract_features(file_path, sr=22050, n_mfcc=20):
    """
    MUST match the feature extraction in train_audio_models.py.
    """
    y, sr = librosa.load(file_path, sr=sr, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    feat = np.concatenate([mfcc_mean, mfcc_std])
    return feat


def predict_audio_emotion(audio_file_path: str) -> Tuple[str, str, str]:
    """
    Returns 3 model predictions: (svm_emotion, nb_emotion, kmeans_emotion)
    """

    feat = extract_features(audio_file_path)
    feat = feat.reshape(1, -1)

    # SVM
    svm_pred_id = svm_model.predict(feat)[0]
    svm_emotion = label_encoder.inverse_transform([svm_pred_id])[0]

    # Naive Bayes
    nb_pred_id = nb_model.predict(feat)[0]
    nb_emotion = label_encoder.inverse_transform([nb_pred_id])[0]

    # KMeans
    cluster_id = int(kmeans_model.predict(feat)[0])
    mapped_label_id = kmeans_cluster_map.get(cluster_id, svm_pred_id)
    kmeans_emotion = label_encoder.inverse_transform([mapped_label_id])[0]

    return svm_emotion, nb_emotion, kmeans_emotion
