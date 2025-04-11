import streamlit as st
import os
import random
import tarfile
import urllib.request
import librosa
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile

st.set_page_config(page_title="Google Speech Commands Explorer", layout="centered")
st.title("🎙️ Google Speech Commands App")
st.write("This app previews random samples from the Google Speech Commands v2 dataset and lets you test the input interface for model inference.")

# ---- SECTION 1: Download and preview dataset ----
st.header("🔍 Dataset Preview")

DATASET_URL = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
DATASET_DIR = Path("speech_commands_data")

@st.cache_resource
def download_and_extract_dataset():
    if not DATASET_DIR.exists():
        os.makedirs(DATASET_DIR, exist_ok=True)
        st.info("Downloading dataset... (only first time)")
        file_path, _ = urllib.request.urlretrieve(DATASET_URL)
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=DATASET_DIR)
        st.success("Dataset downloaded and extracted!")
    return DATASET_DIR

dataset_path = download_and_extract_dataset()

# Pick 5 random command folders (excluding _background_noise_)
all_labels = [d.name for d in dataset_path.iterdir() if d.is_dir() and not d.name.startswith("_")]
sample_labels = random.sample(all_labels, 5)

st.markdown(f"**Displaying random samples from these labels:** `{', '.join(sample_labels)}`")

for label in sample_labels:
    label_path = dataset_path / label
    wav_files = list(label_path.glob("*.wav"))
    if not wav_files:
        continue
    example_file = random.choice(wav_files)
    st.markdown(f"**Label:** `{label}` — File: `{example_file.name}`")
    st.audio(str(example_file), format='audio/wav')

# ---- SECTION 2: Upload audio for inference ----
st.header("🧪 Upload Your Audio")

uploaded_file = st.file_uploader("Upload a WAV audio file for prediction", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    st.success("Audio uploaded successfully!")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        tmpfile.write(uploaded_file.read())
        tmpfile_path = tmpfile.name

    audio, sr = librosa.load(tmpfile_path, sr=None)
    st.write(f"Sample Rate: {sr}, Duration: {len(audio) / sr:.2f}s")

    st.subheader("Waveform")
    fig, ax = plt.subplots()
    ax.plot(audio)
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

    st.subheader("Predicted Label")
    st.warning("⚠️ Model not loaded yet. Prediction will appear here after export.")
else:
    st.info("Please upload an audio file to get started.")
