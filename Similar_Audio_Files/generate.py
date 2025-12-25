import numpy as np
import soundfile as sf
import os

# Create output folder
os.makedirs("synthetic_audio", exist_ok=True)

# Sampling rate
sr = 16000  

# Generate 10 audio files with slightly different frequencies
for i in range(10):
    duration = 3.0  # seconds
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Base frequency around 440 Hz (A4 note), vary slightly
    freq = 440 + i * 50
    waveform = 0.5 * np.sin(2 * np.pi * freq * t)

    filename = f"synthetic_audio/audio_{i+1}.mp3"
    sf.write(filename, waveform, sr, format="MP3")

    print(f"Generated {filename}")
