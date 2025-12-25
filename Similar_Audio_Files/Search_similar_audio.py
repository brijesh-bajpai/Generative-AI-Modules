import os
import torch
import torchaudio
import soundfile as sf
import librosa
import numpy as np
from scipy.spatial.distance import cosine


# -------------------------------
# 1. Feature Extraction Function
# -------------------------------
def extract_features(audio_path):
    
    waveform, sample_rate = librosa.load(audio_path, sr=None, mono=True)

    # Compute Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_mels=64)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    features = log_mel_spec.mean(axis=1)  # average over time
    return features


# -------------------------------
# 2. Store Audio Features
# -------------------------------
def build_database(folder_path):
    database = {}
    for file in os.listdir(folder_path):
        if file.endswith(".wav") or file.endswith(".mp3"):
            path = os.path.join(folder_path, file)
            features = extract_features(path)
            database[file] = features
    return database

# -------------------------------
# 3. Compare Query Audio
# -------------------------------
def find_similar_audio(query_path, database, top_k=3):
    query_features = extract_features(query_path)
    similarities = []

    for file, features in database.items():
        sim = 1 - cosine(query_features, features)  # cosine similarity
        similarities.append((file, sim))

    # Sort by similarity score
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

# -------------------------------
# 4. Example Usage
# -------------------------------
if __name__ == "__main__":
    # Folder containing stored audio files
    audio_folder = "Similar_Audio_Files/synthetic_audio"   # put your audio files here
    os.makedirs(audio_folder, exist_ok=True)

    # Build database
    db = build_database(audio_folder)

    # Path to query audio file
    query_file = "Similar_Audio_Files/audio_1.mp3"   # replace with your uploaded file

    # Find top 3 similar files
    results = find_similar_audio(query_file, db, top_k=3)

    print("Query:", query_file)
    print("Top similar audio files:")
    for fname, score in results:
        print(f"{fname} (similarity: {score:.4f})")