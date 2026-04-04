"""
record_and_predict.py
=====================
Records audio from your microphone → saves as MP3 → runs the full
NLP prediction pipeline → outputs severity: low / medium / high

Requirements (install once)
---------------------------
  pip install sounddevice soundfile pydub pyaudio numpy requests

  Also needs ffmpeg on your system:
    Mac  :  brew install ffmpeg
    Linux:  sudo apt install ffmpeg
    Win  :  https://ffmpeg.org/download.html  (add to PATH)

Files needed in the same folder
--------------------------------
  svm_model.pkl
  normalization_map.json
  normalized_stopwords.json
  normalized_suffixes.json

Usage
-----
  python record_and_predict.py               # press Enter to stop
  python record_and_predict.py --seconds 10  # auto-stop after 10 s
"""

import argparse
import io
import json
import os
import pickle
import re
import sys
import time
import unicodedata
import wave

import numpy as np
import requests

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
API_KEY       = "211d63664dfd4079bb1bea18cea0adef"
BASE_URL      = "https://api.assemblyai.com"
HEADERS       = {"authorization": API_KEY}

OUTPUT_MP3    = "healthvoice.mp3"
SAMPLE_RATE   = 16000   # Hz — good quality for speech
CHANNELS      = 1       # mono
MIN_ROOT_LEN  = 2

MODEL_PATH    = "svm_model.pkl"
NORM_MAP_PATH = "normalization_map.json"
SW_PATH       = "normalized_stopwords.json"
SFX_PATH      = "normalized_suffixes.json"

SEVERITY_EMOJI = {"low": "🟢", "medium": "🟡", "high": "🔴"}
SEVERITY_DESC  = {
    "low":    "Routine — no immediate action needed",
    "medium": "Moderate — monitor closely, consult if worsens",
    "high":   "Urgent — immediate medical attention required",
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 0 — LOAD NLP ASSETS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def load_assets():
    missing = [p for p in [MODEL_PATH, NORM_MAP_PATH, SW_PATH, SFX_PATH]
               if not os.path.exists(p)]
    if missing:
        print("❌  Missing required files:")
        for m in missing:
            print(f"    {m}")
        sys.exit(1)

    model        = pickle.load(open(MODEL_PATH, "rb"))
    norm_map     = json.load(open(NORM_MAP_PATH, encoding="utf-8"))
    stopword_set = set(json.load(open(SW_PATH, encoding="utf-8"))["stopwords"])
    suffix_list  = json.load(open(SFX_PATH,  encoding="utf-8"))["suffixes"]
    return model, norm_map, stopword_set, suffix_list


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NLP PIPELINE  (identical to training)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def build_nlp(norm_map, stopword_set, suffix_list):
    sorted_keys = sorted(norm_map.keys(), key=len, reverse=True)
    protected   = set("".join(norm_map.values()))

    def normalize(text):
        text = unicodedata.normalize("NFC", text)
        for key in sorted_keys:
            rep = norm_map[key]
            if rep == "" and key in protected:
                continue
            if key in text:
                text = text.replace(key, rep)
        return " ".join(text.split())

    def tokenize(text):
        tokens = []
        for t in text.split():
            parts = re.split(r"[–\-]", t)
            tokens.extend(p for p in parts if p)
        return tokens

    def remove_stopwords(tokens):
        return [t for t in tokens if t not in stopword_set]

    def strip_suffix(token):
        for sfx in suffix_list:
            if token.endswith(sfx) and len(token) - len(sfx) >= MIN_ROOT_LEN:
                return token[: -len(sfx)], sfx
        return token, None

    def stem_tokens(tokens):
        return [strip_suffix(t)[0] for t in tokens]

    def preprocess(raw_text):
        norm   = normalize(raw_text)
        toks   = tokenize(norm)
        toks   = remove_stopwords(toks)
        toks   = stem_tokens(toks)
        return norm, toks, " ".join(toks)

    return preprocess


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 1 — RECORD AUDIO
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def record_audio(seconds=None):
    """
    Records from the default microphone.
    If seconds is given → auto-stop after that duration.
    Otherwise         → press Enter to stop.
    """
    try:
        import sounddevice as sd
        import soundfile as sf
    except ImportError:
        print("❌  Missing libraries. Run:")
        print("      pip install sounddevice soundfile pydub numpy")
        sys.exit(1)

    frames = []

    if seconds:
        print(f"\n🎙️  Recording for {seconds} second(s)... speak now!\n")
        audio = sd.rec(int(seconds * SAMPLE_RATE),
                       samplerate=SAMPLE_RATE,
                       channels=CHANNELS,
                       dtype="int16")
        # live countdown
        for remaining in range(seconds, 0, -1):
            print(f"    ⏱  {remaining}s remaining...", end="\r", flush=True)
            time.sleep(1)
        sd.wait()
        print("\n⏹️  Recording stopped.")
        audio_data = audio
    else:
        print("\n🎙️  Recording started — speak now!")
        print("    Press  Enter  to stop.\n")

        recording = []
        stop_flag = [False]

        def callback(indata, frame_count, time_info, status):
            if not stop_flag[0]:
                recording.append(indata.copy())

        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            callback=callback,
        )
        with stream:
            input()   # blocks until Enter
            stop_flag[0] = True

        print("⏹️  Recording stopped.")
        if not recording:
            print("❌  No audio captured.")
            sys.exit(1)
        audio_data = np.concatenate(recording, axis=0)

    # ── Save as WAV first, then convert to MP3 via pydub ──────────────
    wav_path = OUTPUT_MP3.replace(".mp3", "_tmp.wav")
    import soundfile as sf
    sf.write(wav_path, audio_data, SAMPLE_RATE)
    print(f"💾  WAV saved: {wav_path}")

    # Convert WAV → MP3
    try:
        from pydub import AudioSegment
        AudioSegment.from_wav(wav_path).export(OUTPUT_MP3, format="mp3", bitrate="128k")
        os.remove(wav_path)
        print(f"🎵  MP3 saved: {OUTPUT_MP3}  ({os.path.getsize(OUTPUT_MP3):,} bytes)")
    except Exception as e:
        print(f"⚠️   Could not convert to MP3 ({e}). Using WAV instead.")
        OUTPUT_MP3_actual = wav_path
        return wav_path

    return OUTPUT_MP3


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 2 — UPLOAD + TRANSCRIBE (AssemblyAI)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def transcribe(audio_path):
    print(f"\n⏫  Uploading '{audio_path}'...")
    with open(audio_path, "rb") as f:
        resp = requests.post(BASE_URL + "/v2/upload", headers=HEADERS, data=f)
    resp.raise_for_status()
    audio_url = resp.json()["upload_url"]
    print("✅  Upload complete.")

    data = {
        "audio_url":          audio_url,
        "speech_models":      ["universal-2"],
        "language_detection": True,
        "language_detection_options": {
            "expected_languages": ["ne"],
            "fallback_language":  "ne",
        },
    }
    print("📤  Requesting Nepali transcription...")
    resp = requests.post(
        BASE_URL + "/v2/transcript",
        json=data,
        headers={**HEADERS, "content-type": "application/json"},
    )
    if not resp.ok:
        print(f"❌  {resp.status_code}: {resp.text}")
        resp.raise_for_status()

    tid = resp.json()["id"]
    print(f"🆔  Transcript ID: {tid}")

    polling_url = f"{BASE_URL}/v2/transcript/{tid}"
    print("⏳  Transcribing", end="", flush=True)
    while True:
        result = requests.get(polling_url, headers=HEADERS).json()
        if result["status"] == "completed":
            print("\n✅  Transcription done.")
            return result["text"], result.get("language_code", "unknown")
        elif result["status"] == "error":
            raise RuntimeError(f"❌  Transcription failed: {result['error']}")
        print(".", end="", flush=True)
        time.sleep(3)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    parser = argparse.ArgumentParser(description="Record Nepali audio and predict severity.")
    parser.add_argument("--seconds", type=int, default=None,
                        help="Auto-stop recording after N seconds. Omit to press Enter to stop.")
    args = parser.parse_args()

    print("=" * 60)
    print("  HealthVoice — Nepali Audio Severity Predictor")
    print("=" * 60)

    # Load assets
    print("\n📦  Loading model and NLP assets...")
    model, norm_map, stopword_set, suffix_list = load_assets()
    preprocess = build_nlp(norm_map, stopword_set, suffix_list)
    print("✅  Ready.\n")

    # Record
    audio_path = record_audio(seconds=args.seconds)

    # Transcribe
    raw_text, detected_lang = transcribe(audio_path)

    # NLP preprocess
    norm_text, tokens_stemmed, model_input = preprocess(raw_text)

    # Predict
    prediction = model.predict([model_input])[0]

    # ── Print result ──────────────────────────────────────────
    SEP = "=" * 60
    print(f"\n{SEP}")
    print(f"🌐  Detected language   : {detected_lang}")
    print(f"\n📝  Raw transcription   :\n    {raw_text}")
    print(f"\n🔤  Normalized          :\n    {norm_text}")
    print(f"\n🪙  Tokens (stemmed)    :\n    {tokens_stemmed}")
    print(f"\n{SEP}")
    print(f"🏥  SEVERITY PREDICTION :  "
          f"{SEVERITY_EMOJI[prediction]}  {prediction.upper()}")
    print(f"    {SEVERITY_DESC[prediction]}")
    print(SEP)

    # Save result
    result = {
        "audio_file":        audio_path,
        "detected_language": detected_lang,
        "raw_transcription": raw_text,
        "normalized_text":   norm_text,
        "tokens_stemmed":    tokens_stemmed,
        "model_input":       model_input,
        "prediction":        prediction,
        "description":       SEVERITY_DESC[prediction],
    }
    out = "prediction_result.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n💾  Result saved to '{out}'")


if __name__ == "__main__":
    main()