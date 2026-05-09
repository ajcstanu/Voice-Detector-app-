**Voice Detector** app!! 

---

## ⚙️ Setup (run once)

```bash
pip install librosa sounddevice numpy scipy
```

> On Linux you may also need: `sudo apt install portaudio19-dev`

---

## ▶️ Run the app

```bash
python voice_detector.py
```

---

## 🧠 What it detects

| Voice Type | Detection Method |
|---|---|
| 👨 **Man** | F0 < 165 Hz + age estimated from pitch drop |
| 👩 **Woman** | F0 165–255 Hz + spectral centroid |
| 👦 **Boy** | F0 > 200 Hz + high ZCR + narrow centroid |
| 👧 **Girl** | F0 > 250 Hz + high ZCR + wide centroid |
| 🐦 **Bird** | F0 > 1500 Hz or spectral centroid > 4000 Hz |
| 🐕 **Dog** | Low voiced ratio, percussive, low-mid pitch |
| 🐈 **Cat** | Mid harmonic vocalisation |
| 🎵 **Music** | Very high harmonic ratio, steady pitch |
| ❓ **Unknown** | Doesn't fit any category |

---

## 🎯 Age Estimation Logic

Uses **fundamental frequency (F0)** + **spectral centroid** + **zero-crossing rate**:
- Children have higher F0 (200–400 Hz) and higher ZCR
- Young adults have strong, mid-range F0
- Older voices show lower F0, lower ZCR, and narrower spectral bandwidth

---

| Section | Content |
|---|---|
| **1. Overview** | What the app does and how it works |
| **2. Features** | Detection categories table, age estimation logic, confidence scores |
| **3. Requirements** | System requirements + Python packages table |
| **4. Installation** | Step-by-step setup for Windows, macOS, and Linux |
| **5. Usage Guide** | Menu options, file analysis, sample terminal output |
| **6. How It Works** | Signal pipeline + all 8 acoustic features explained |
| **7. Tips** | Environment tips, recording duration, known limitations |
| **8. Troubleshooting** | Common errors with causes and fixes |
| **9. File Structure** | Project folder layout |
| **10. Cheat Sheet** | Quick-reference commands table |

## 📋 Menu Options
1. **Record 3 sec** —> quick detect
2. **Custom duration** —> 1 to 10 seconds
3. **Analyze a file** —> .wav or .mp3
4. **Continuous mode** —> keeps looping
5. **Exit**
-------

