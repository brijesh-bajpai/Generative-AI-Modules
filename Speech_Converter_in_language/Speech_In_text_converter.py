import os
import sys
from typing import Tuple

import speech_recognition as sr
from deep_translator import GoogleTranslator
from gtts import gTTS
from pydub import AudioSegment

# Five output language choices
OUTPUT_LANGUAGES = {
    "1": ("English", "en"),
    "2": ("Hindi", "hi"),
    "3": ("Spanish", "es"),
    "4": ("French", "fr"),
    "5": ("German", "de"),
}

def choose_output_language() -> Tuple[str, str]:
    print("Select output language:")
    for k, (name, _) in OUTPUT_LANGUAGES.items():
        print(f"{k}. {name}")
    choice = input("Enter choice number: ").strip()
    return OUTPUT_LANGUAGES.get(choice, ("English", "en"))

def load_audio_as_wav(input_path: str) -> str:
    """Convert audio file to temp WAV for recognition."""
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Audio file not found: {input_path}")

    ext = os.path.splitext(input_path)[1].lower()
    temp_wav = "temp_input.wav"

    if ext == ".wav":
        audio = AudioSegment.from_wav(input_path)
    else:
        audio = AudioSegment.from_file(input_path)

    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    audio.export(temp_wav, format="wav")
    return temp_wav

def transcribe_audio(wav_path: str, language_hint: str = "en-US") -> str:
    """Transcribe speech from WAV using Google Web Speech API."""
    r = sr.Recognizer()
    with sr.AudioFile(wav_path) as source:
        audio = r.record(source)
    try:
        text = r.recognize_google(audio, language=language_hint)
        print(f"Transcribed text: {text}")
        return text
    except sr.UnknownValueError:
        print("Could not understand the audio.")
        return ""
    except sr.RequestError as e:
        print(f"Speech service error: {e}")
        return ""

def translate_text(text: str, dest_lang_code: str) -> str:
    translated = GoogleTranslator(source='auto', target=dest_lang_code).translate(text)
    print(f"Translated text ({dest_lang_code}): {translated}")
    return translated

def synthesize_speech(text: str, lang_code: str, out_path: str = "output.mp3") -> str:
    tts = gTTS(text=text, lang=lang_code)
    tts.save(out_path)
    print(f"Saved synthesized speech: {out_path}")
    return out_path

def open_audio(output_path: str):
    try:
        os.system(f'start "" "{output_path}"')  # Windows
    except Exception:
        print(f"Play the file manually: {output_path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python translator_app.py <input_audio_path> [recognition_language_hint]")
        print("Examples:")
        print("  python translator_app.py input.wav en-US")
        print("  python translator_app.py speech.mp3 hi-IN")
        sys.exit(1)

    input_path = sys.argv[1]
    rec_lang_hint = sys.argv[2] if len(sys.argv) >= 3 else "en-US"

    # 1) Choose output language
    out_lang_name, out_lang_code = choose_output_language()

    # 2) Convert input to WAV
    temp_wav = load_audio_as_wav(input_path)

    # 3) Transcribe
    spoken_text = transcribe_audio(temp_wav, language_hint=rec_lang_hint)
    if not spoken_text:
        print("No transcription produced. Exiting.")
        sys.exit(2)

    # 4) Translate
    translated = translate_text(spoken_text, dest_lang_code=out_lang_code)

    # 5) Synthesize
    output_mp3 = f"translated_{out_lang_code}.mp3"
    synthesize_speech(translated, lang_code=out_lang_code, out_path=output_mp3)

    # 6) Play
    open_audio(output_mp3)

    # Cleanup
    if os.path.exists(temp_wav):
        try:
            os.remove(temp_wav)
        except Exception:
            pass

if __name__ == "__main__":
    main()