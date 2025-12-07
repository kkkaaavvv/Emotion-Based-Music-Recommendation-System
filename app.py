# app.py

import io
import os

import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import streamlit as st
import streamlit.components.v1 as components

# 🔊 AUDIO MODELS
# If predict_audio.py is inside utils/  (as in your screenshot), this import is correct.
# If it's in the project root instead, change this line to:
#   from predict_audio import predict_audio_emotion
from utils.predict_audio import predict_audio_emotion


# -------------------------------------------------
# 1. Your Spotify playlists (same as before)
# -------------------------------------------------
PLAYLISTS = {
    "happy": {
        "name": "Golden Hour Glow ✨",
        "url": "https://open.spotify.com/playlist/4RKOpn4egd37fxakJSchF0?si=971f45c364664ed5",
        "cover": "https://i.scdn.co/image/ab67616d0000b273f9da4ad6a79a3fd7c5f3e34d",
    },
    "sad": {
        "name": "Midnight in My Mind 🌙",
        "url": "https://open.spotify.com/playlist/3yWQSi1Io43q26w57Sza0o?si=f4605f9a92604178",
        "cover": "https://i.scdn.co/image/ab67616d0000b2735a71df52439f0c1b15f8f8d8",
    },
    "calm": {
        "name": "Pastel Peace 🍃",
        "url": "https://open.spotify.com/playlist/0EedJsRbtKmRKeRoodUpHX?si=22a5d08da0fb42c9",
        "cover": "https://i.scdn.co/image/ab67616d0000b273b6bc447ddb5a0e67e0c9383c",
    },
    "anger": {
        "name": "R3V3NG3 L00P 🔥",
        "url": "https://open.spotify.com/playlist/0x3nOzZ7axAdryQxxujPjd?si=38617933d11941da",
        "cover": "https://i.scdn.co/image/ab67616d0000b2730ce4e3d0f5c1a7109a2c8e9b",
    },
    "fear": {
        "name": "Velvet Fear 🩸",
        "url": "https://open.spotify.com/playlist/3QNuYmHA8rtvgGPETPvUZo?si=2894b574b7874b2f",
        "cover": "https://i.scdn.co/image/ab67616d0000b273cd8f438db089d1dcef2fdf42",
    },
    "surprise": {
        "name": "Fairy Dust ✨",
        "url": "https://open.spotify.com/playlist/2dhd7H9WG5z0MqN4dw98UU?si=6ba8aba39a2d4574",
        "cover": "https://i.scdn.co/image/ab67616d0000b2734cbc2f1521f1d7a60f7f8012",
    },
    "neutral": {
        "name": "Vanilla Mornings ☀️",
        "url": "https://open.spotify.com/playlist/4B3NYLiLXe9iA1x2qBeB5v?si=c930614bad1a4a25",
        "cover": "https://i.scdn.co/image/ab67616d0000b2737f9326b6c4b6f0b8e9cd2dba",
    },
}


# -------------------------------------------------
# 2. Camera Emotion Model
# -------------------------------------------------
MODEL = load_model("models/camera_cnn/trained_model/model.h5")
FACE_DETECTOR = cv2.CascadeClassifier("utils/haarcascade_frontalface_default.xml")
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def detect_emotion_from_frame(frame: np.ndarray):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_DETECTOR.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi = gray[y:y + h, x:x + w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float32") / 255.0
        roi = np.expand_dims(roi, axis=-1)
        roi = np.expand_dims(roi, axis=0)

        preds = MODEL.predict(roi)[0]
        return EMOTIONS[int(np.argmax(preds))]

    return None


# -------------------------------------------------
# 3. Playlist helpers
# -------------------------------------------------
def make_embed_url(url: str) -> str:
    return url.replace(
        "open.spotify.com/playlist/",
        "open.spotify.com/embed/playlist/",
    )


def get_playlist_for(emotion: str):
    emotion = (emotion or "").lower().strip()

    alias = {
        "angry": "anger",
        "disgust": "anger",
    }
    key = alias.get(emotion, emotion)

    playlist = PLAYLISTS.get(key, PLAYLISTS["neutral"]).copy()
    playlist["embed_url"] = make_embed_url(playlist["url"])
    return playlist


def convert_streamlit_image_to_frame(image):
    bytes_data = image.getvalue()
    pil_img = Image.open(io.BytesIO(bytes_data)).convert("RGB")
    frame_rgb = np.array(pil_img)
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)


def pick_final_emotion_from_audio(e1: str, e2: str, e3: str) -> str:
    """
    Majority vote over (SVM, NB, KMeans).
    If all three are different, trust SVM (e1).
    """
    preds = [p.lower().strip() for p in (e1, e2, e3)]
    best = max(set(preds), key=preds.count)
    if preds.count(best) == 1:
        best = preds[0]
    return best


# -------------------------------------------------
# 4. Streamlit App
# -------------------------------------------------
st.set_page_config(page_title="Emotion → Spotify", page_icon="🎧", layout="centered")
st.title("🎭 Emotion → 🎧 Spotify Playlist")
st.write("Camera for **face emotion**, audio for **voice emotion**. Two vibes, two playlists.")


# ---- Section 1: Camera ----
st.markdown("## 📷 1️⃣ Capture your face")
img = st.camera_input("Look at the camera and click **Take Photo**")

if img:
    frame = convert_streamlit_image_to_frame(img)
    emotion = detect_emotion_from_frame(frame)

    st.markdown("---")

    if emotion:
        playlist = get_playlist_for(emotion)

        st.markdown(f"### 🧠 Face Emotion: **{emotion.capitalize()}**")

        st.markdown(
            f"""
            <div style="text-align: center; margin-top: 10px; margin-bottom: 10px;">
                <img src="{playlist['cover']}" style="width:180px; border-radius:15px; box-shadow: 0 8px 20px rgba(0,0,0,0.25);">
                <h3 style="margin-top: 15px; margin-bottom: 5px;">{playlist['name']}</h3>
                <a href="{playlist['url']}" target="_blank">
                    <button style="
                        background-color:#1DB954;
                        padding:10px 24px;
                        color:white;
                        border:none;
                        border-radius:999px;
                        font-size:16px;
                        font-weight:600;
                        cursor:pointer;
                        margin-top:8px;
                    ">Open Face-based Playlist</button>
                </a>
            </div>
            """,
            unsafe_allow_html=True,
        )

        components.iframe(
            playlist["embed_url"],
            width=600,
            height=400,
            scrolling=True,
        )
    else:
        st.warning("No face detected — try again 👀")
else:
    st.info("Take a photo to detect your emotion and show your playlist!")


# ---- Section 2: Audio ----
st.markdown("## 🔊 2️⃣ Upload your voice")

audio_file = st.file_uploader(
    "Upload a short audio clip (you can even use one from audio_dataset)",
    type=["wav", "mp3", "ogg", "flac"],
    accept_multiple_files=False,
)

if audio_file is not None:
    temp_audio_path = "temp_uploaded_audio.wav"
    with open(temp_audio_path, "wb") as f:
        f.write(audio_file.read())

    # Use your 3 audio models (SVM, NB, KMeans)
    svm_emotion, nb_emotion, kmeans_emotion = predict_audio_emotion(temp_audio_path)
    final_audio_emotion = pick_final_emotion_from_audio(
        svm_emotion, nb_emotion, kmeans_emotion
    )

    st.markdown("---")
    st.markdown(f"### 🎧 Audio Emotion: **{final_audio_emotion.capitalize()}**")

    with st.expander("See individual model predictions"):
        st.write("SVM:", svm_emotion)
        st.write("Naive Bayes:", nb_emotion)
        st.write("KMeans:", kmeans_emotion)

    audio_playlist = get_playlist_for(final_audio_emotion)

    st.markdown(
        f"""
        <div style="text-align: center; margin-top: 10px; margin-bottom: 10px;">
            <img src="{audio_playlist['cover']}" style="width:180px; border-radius:15px; box-shadow: 0 8px 20px rgba(0,0,0,0.25);">
            <h3 style="margin-top: 15px; margin-bottom: 5px;">{audio_playlist['name']}</h3>
            <a href="{audio_playlist['url']}" target="_blank">
                <button style="
                    background-color:#1DB954;
                    padding:10px 24px;
                    color:white;
                    border:none;
                    border-radius:999px;
                    font-size:16px;
                    font-weight:600;
                    cursor:pointer;
                    margin-top:8px;
                ">Open Audio-based Playlist</button>
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    components.iframe(
        audio_playlist["embed_url"],
        width=600,
        height=400,
        scrolling=True,
    )
else:
    st.info("Upload an audio clip to get another playlist based on your voice 🎶")

