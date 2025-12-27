from gtts import gTTS
from pydub import AudioSegment

# Your short speech text
speech_text = "Good evening everyone. Today we are learning how technology can connect people across languages. My children not focusing on study. Should I punish him?"

# Step 1: Generate MP3 using gTTS
tts = gTTS(text=speech_text, lang='en')
tts.save("speech.mp3")

# Step 2: Convert MP3 to WAV using pydub
audio = AudioSegment.from_mp3("speech.mp3")
audio.export("speech.wav", format="wav")

print("✅ Speech saved as speech.wav")