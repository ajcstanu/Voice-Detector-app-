"""
Voice Detector - Detects human (man/woman/boy/girl + age), animals, birds, etc.
Requirements: pip install librosa numpy sounddevice scipy scikit-learn pyaudio
"""

import numpy as np
import librosa
import sounddevice as sd
import scipy.signal
import warnings
import sys
import time
import threading

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
SAMPLE_RATE = 22050       # Hz
DURATION    = 3           # seconds per recording
CHANNELS    = 1

# ─────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────
def extract_features(audio: np.ndarray, sr: int) -> dict:
    """Extract acoustic features from audio signal."""
    # Remove silence
    audio, _ = librosa.effects.trim(audio, top_db=20)
    if len(audio) < sr * 0.2:
        return None

    features = {}

    # 1. Fundamental frequency (pitch)
    f0, voiced_flag, _ = librosa.pyin(
        audio, fmin=librosa.note_to_hz("C1"),
        fmax=librosa.note_to_hz("C8"), sr=sr
    )
    voiced_f0 = f0[voiced_flag] if voiced_flag is not None else np.array([])
    features["f0_mean"]   = float(np.nanmean(voiced_f0))   if len(voiced_f0) > 0 else 0
    features["f0_std"]    = float(np.nanstd(voiced_f0))    if len(voiced_f0) > 0 else 0
    features["f0_median"] = float(np.nanmedian(voiced_f0)) if len(voiced_f0) > 0 else 0

    # 2. MFCCs (timbre)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    features["mfcc_mean"] = float(np.mean(mfccs[0]))
    features["mfcc_std"]  = float(np.std(mfccs[0]))
    for i in range(1, 6):
        features[f"mfcc_{i}_mean"] = float(np.mean(mfccs[i]))

    # 3. Spectral features
    spectral_centroid  = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
    spectral_rolloff   = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
    zcr                = librosa.feature.zero_crossing_rate(audio)[0]

    features["spectral_centroid_mean"]  = float(np.mean(spectral_centroid))
    features["spectral_bandwidth_mean"] = float(np.mean(spectral_bandwidth))
    features["spectral_rolloff_mean"]   = float(np.mean(spectral_rolloff))
    features["zcr_mean"]                = float(np.mean(zcr))
    features["zcr_std"]                 = float(np.std(zcr))

    # 4. Chroma
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    features["chroma_mean"] = float(np.mean(chroma))
    features["chroma_std"]  = float(np.std(chroma))

    # 5. Energy / RMS
    rms = librosa.feature.rms(y=audio)[0]
    features["rms_mean"] = float(np.mean(rms))
    features["rms_std"]  = float(np.std(rms))

    # 6. Voiced ratio
    features["voiced_ratio"] = (
        float(np.sum(voiced_flag) / len(voiced_flag))
        if voiced_flag is not None and len(voiced_flag) > 0 else 0
    )

    # 7. Tempo / rhythm
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    features["tempo"] = float(tempo)

    # 8. Harmonic vs percussive ratio
    harmonic, percussive = librosa.effects.hpss(audio)
    features["harmonic_ratio"] = (
        float(np.sum(harmonic**2) / (np.sum(audio**2) + 1e-10))
    )

    return features


# ─────────────────────────────────────────────
# CLASSIFICATION ENGINE
# ─────────────────────────────────────────────
def estimate_human_age(f0_mean: float, f0_std: float,
                        spectral_centroid: float, zcr: float,
                        gender: str) -> tuple[int, int]:
    """
    Estimate age range from acoustic features.
    Returns (min_age, max_age).
    """
    if gender == "boy":
        if f0_mean > 300:
            return (3, 7)
        elif f0_mean > 250:
            return (7, 11)
        elif f0_mean > 200:
            return (11, 14)
        else:
            return (14, 17)

    elif gender == "girl":
        if f0_mean > 320:
            return (3, 7)
        elif f0_mean > 270:
            return (7, 11)
        elif f0_mean > 230:
            return (11, 15)
        else:
            return (15, 17)

    elif gender == "man":
        # Male adult pitch + formant cues
        if f0_mean > 140:
            return (18, 25)
        elif f0_mean > 115:
            if spectral_centroid < 1200:
                return (45, 65)
            return (25, 45)
        else:
            if zcr < 0.04:
                return (60, 85)
            return (45, 70)

    else:  # woman
        if f0_mean > 250:
            return (18, 28)
        elif f0_mean > 210:
            if spectral_centroid < 1500:
                return (40, 60)
            return (28, 45)
        else:
            if zcr < 0.05:
                return (58, 80)
            return (45, 65)


def classify_voice(features: dict) -> dict:
    """
    Rule-based classifier using acoustic features.
    Returns a dict with category, sub-type, confidence, description.
    """
    if features is None:
        return {"category": "unknown", "label": "No voice detected",
                "confidence": 0, "description": "Signal too short or silent"}

    f0       = features["f0_mean"]
    f0_std   = features["f0_std"]
    sc       = features["spectral_centroid_mean"]
    zcr      = features["zcr_mean"]
    zcr_std  = features["zcr_std"]
    voiced   = features["voiced_ratio"]
    harmonic = features["harmonic_ratio"]
    rms      = features["rms_mean"]
    bw       = features["spectral_bandwidth_mean"]
    rolloff  = features["spectral_rolloff_mean"]

    # ── SILENCE / NOISE ────────────────────────────────────────────────
    if rms < 0.005:
        return {"category": "noise", "label": "Silence / No Sound",
                "confidence": 95, "description": "No significant audio detected"}

    # ── BIRD ──────────────────────────────────────────────────────────
    # Birds: very high pitch (2000-8000 Hz), high ZCR, narrow bandwidth,
    #        short rapid bursts, high harmonic ratio
    if f0 > 1500 or (sc > 4000 and zcr > 0.15 and harmonic > 0.6):
        conf = min(95, int(50 + (f0 / 100) + (zcr * 100)))
        return {
            "category": "animal", "label": "🐦 Bird",
            "confidence": min(conf, 95),
            "description": (
                f"High-pitched vocalisation detected (F0≈{f0:.0f} Hz). "
                "Likely a bird based on spectral characteristics."
            )
        }

    # ── ANIMAL (non-bird) ─────────────────────────────────────────────
    # Low voiced ratio but lots of energy, or very low rumble / bark
    if voiced < 0.2 and rms > 0.02 and f0 < 600:
        # Dog bark: sharp, percussive, low-mid pitch
        if zcr > 0.12 and harmonic < 0.4:
            return {"category": "animal", "label": "🐕 Dog (Bark/Growl)",
                    "confidence": 78,
                    "description": "Percussive, low-voiced burst consistent with dog vocalisation."}
        # Cat meow: moderate pitch, more harmonic
        if 300 < f0 < 800 and harmonic > 0.4:
            return {"category": "animal", "label": "🐈 Cat (Meow/Purr)",
                    "confidence": 72,
                    "description": "Mid-range harmonic vocalisation consistent with cat sounds."}
        # Generic animal
        return {"category": "animal", "label": "🐾 Animal (Unknown)",
                "confidence": 60,
                "description": "Non-human, non-bird vocalisation detected."}

    # ── MUSICAL INSTRUMENT / SINGING ─────────────────────────────────
    if harmonic > 0.75 and f0_std < 30 and zcr < 0.05:
        return {"category": "sound", "label": "🎵 Music / Instrument",
                "confidence": 70,
                "description": "Highly harmonic, steady-pitch signal — likely music or singing."}

    # ── HUMAN VOICE ───────────────────────────────────────────────────
    # Typical human speech: voiced > 0.35, F0 in 80–400 Hz
    if voiced >= 0.25 and 60 <= f0 <= 600:

        # ── CHILD VOICE ─────────────────────────────────────────────
        # Children: F0 > 200 Hz, higher SC, higher ZCR
        if f0 > 200 and sc > 2200 and zcr > 0.07:
            gender = "girl" if f0 > 250 else "boy"
            age_lo, age_hi = estimate_human_age(f0, f0_std, sc, zcr, gender)
            icon = "👧" if gender == "girl" else "👦"
            label = f"{icon} {'Girl' if gender=='girl' else 'Boy'} (~{age_lo}–{age_hi} yrs)"
            conf = min(90, int(55 + voiced * 40 + (f0 / 20)))
            return {
                "category": "human", "label": label,
                "confidence": conf,
                "description": (
                    f"F0≈{f0:.0f} Hz, spectral centroid={sc:.0f} Hz — "
                    f"child voice characteristics detected. Estimated age {age_lo}–{age_hi} years."
                )
            }

        # ── ADULT MALE ───────────────────────────────────────────────
        if f0 <= 165:
            age_lo, age_hi = estimate_human_age(f0, f0_std, sc, zcr, "man")
            conf = min(92, int(60 + voiced * 35))
            return {
                "category": "human",
                "label": f"👨 Man (~{age_lo}–{age_hi} yrs)",
                "confidence": conf,
                "description": (
                    f"F0≈{f0:.0f} Hz — typical adult male fundamental. "
                    f"Estimated age {age_lo}–{age_hi} years."
                )
            }

        # ── ADULT FEMALE ─────────────────────────────────────────────
        if 165 < f0 <= 255 and sc < 3000:
            age_lo, age_hi = estimate_human_age(f0, f0_std, sc, zcr, "woman")
            conf = min(90, int(58 + voiced * 35))
            return {
                "category": "human",
                "label": f"👩 Woman (~{age_lo}–{age_hi} yrs)",
                "confidence": conf,
                "description": (
                    f"F0≈{f0:.0f} Hz — typical adult female fundamental. "
                    f"Estimated age {age_lo}–{age_hi} years."
                )
            }

        # ── AMBIGUOUS HUMAN ──────────────────────────────────────────
        return {
            "category": "human",
            "label": "🧑 Human Voice (uncertain gender)",
            "confidence": 65,
            "description": (
                f"F0≈{f0:.0f} Hz, voiced ratio={voiced:.2f}. "
                "Ambiguous gender/age; try a clearer recording."
            )
        }

    # ── GENERAL NOISE / UNKNOWN ────────────────────────────────────────
    return {
        "category": "unknown", "label": "❓ Unknown Sound",
        "confidence": 40,
        "description": (
            f"F0≈{f0:.0f} Hz, voiced={voiced:.2f}, ZCR={zcr:.3f}. "
            "Could not classify — background noise or unclear signal."
        )
    }


# ─────────────────────────────────────────────
# AUDIO RECORDING
# ─────────────────────────────────────────────
def record_audio(duration: float = DURATION, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Record audio from default microphone."""
    print(f"\n🎙️  Recording for {duration} seconds... Speak now!")
    audio = sd.rec(int(duration * sr), samplerate=sr,
                   channels=CHANNELS, dtype="float32")
    # Animate countdown
    for remaining in range(duration, 0, -1):
        sys.stdout.write(f"\r   ⏳ {remaining}s remaining...   ")
        sys.stdout.flush()
        time.sleep(1)
    sd.wait()
    print("\r   ✅ Recording complete!         ")
    return audio.flatten()


# ─────────────────────────────────────────────
# DISPLAY RESULT
# ─────────────────────────────────────────────
def display_result(result: dict, features: dict):
    """Pretty-print classification result."""
    bar_len = 30
    filled  = int(bar_len * result["confidence"] / 100)
    bar     = "█" * filled + "░" * (bar_len - filled)

    print("\n" + "=" * 55)
    print("  🔬 VOICE ANALYSIS RESULT")
    print("=" * 55)
    print(f"  Detected  : {result['label']}")
    print(f"  Confidence: [{bar}] {result['confidence']}%")
    print(f"  Details   : {result['description']}")
    print("-" * 55)

    if features:
        print(f"  📊 Acoustic Features:")
        print(f"     Fundamental Freq : {features['f0_mean']:.1f} Hz")
        print(f"     Spectral Centroid: {features['spectral_centroid_mean']:.1f} Hz")
        print(f"     Zero-Cross Rate  : {features['zcr_mean']:.4f}")
        print(f"     Voiced Ratio     : {features['voiced_ratio']:.2f}")
        print(f"     Harmonic Ratio   : {features['harmonic_ratio']:.2f}")
        print(f"     RMS Energy       : {features['rms_mean']:.5f}")
    print("=" * 55)


# ─────────────────────────────────────────────
# LOAD FROM FILE
# ─────────────────────────────────────────────
def analyze_file(filepath: str):
    """Analyze an existing audio file."""
    print(f"\n📂 Loading: {filepath}")
    try:
        audio, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
        print(f"   Duration: {len(audio)/sr:.2f}s  |  Sample Rate: {sr} Hz")
        features = extract_features(audio, sr)
        result   = classify_voice(features)
        display_result(result, features)
    except Exception as e:
        print(f"❌ Error loading file: {e}")


# ─────────────────────────────────────────────
# CONTINUOUS MONITORING MODE
# ─────────────────────────────────────────────
def continuous_mode():
    """Continuously record and classify."""
    print("\n🔄  Continuous Mode — press Ctrl+C to stop\n")
    try:
        while True:
            audio    = record_audio(duration=3)
            features = extract_features(audio, SAMPLE_RATE)
            result   = classify_voice(features)
            display_result(result, features)
            print("\n⏸  Pause 1s before next capture...")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n👋 Stopped continuous mode.")


# ─────────────────────────────────────────────
# MAIN MENU
# ─────────────────────────────────────────────
BANNER = r"""
  ╔══════════════════════════════════════════════╗
  ║       🎙️  VOICE DETECTOR & CLASSIFIER       ║
  ║  Detects: Man · Woman · Boy · Girl · Animal  ║
  ║           Bird · Music · Noise + Age Est.    ║
  ╚══════════════════════════════════════════════╝
"""

def main():
    print(BANNER)
    print("  Built with: librosa · sounddevice · numpy")
    print("  ─────────────────────────────────────────\n")

    while True:
        print("  MENU:")
        print("  [1] 🎙️  Record & Detect (3 seconds)")
        print("  [2] 🎙️  Record & Detect (custom duration)")
        print("  [3] 📂  Analyze audio file (.wav / .mp3)")
        print("  [4] 🔄  Continuous mode (auto-loop)")
        print("  [5] ❌  Exit")
        print()

        choice = input("  Enter choice [1-5]: ").strip()

        if choice == "1":
            audio    = record_audio(duration=3)
            features = extract_features(audio, SAMPLE_RATE)
            result   = classify_voice(features)
            display_result(result, features)

        elif choice == "2":
            try:
                dur = float(input("  Duration in seconds (1–10): ").strip())
                dur = max(1, min(10, dur))
            except ValueError:
                dur = 3
            audio    = record_audio(duration=dur)
            features = extract_features(audio, SAMPLE_RATE)
            result   = classify_voice(features)
            display_result(result, features)

        elif choice == "3":
            path = input("  Enter file path: ").strip().strip('"')
            analyze_file(path)

        elif choice == "4":
            continuous_mode()

        elif choice == "5":
            print("\n👋 Goodbye!\n")
            sys.exit(0)

        else:
            print("  ⚠️  Invalid choice, try again.\n")

        print()  # spacing


if __name__ == "__main__":
    # Quick dependency check
    missing = []
    for pkg in ["librosa", "sounddevice", "numpy", "scipy"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print("❌ Missing packages:", ", ".join(missing))
        print("   Run:  pip install " + " ".join(missing))
        sys.exit(1)

    main()
