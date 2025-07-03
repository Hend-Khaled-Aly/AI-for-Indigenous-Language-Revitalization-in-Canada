# 🪶 Cree Language Learning App

This project is a Streamlit-based application designed to support the revitalization of the Cree language. It provides users with tools to translate between Cree and English, practice through exercises, and match spoken audio to Cree words — using both text-based and audio-based machine learning models.

---

## 📁 Project Structure

```
root/
│
├── data/
│   ├── audio/                 # MP3 audio files (975 Cree words)
│   ├── wav/                   # Same 975 audio files converted to WAV
│   ├── raw/                   # Raw dataset: words + translation + audio URL
│   │   └── Dataset-Words_Translation_URLS.csv
│   ├── cleaned/               # Preprocessed and cleaned datasets
│       ├── cree_dataset.csv / .json              # Final processed dataset
│       ├── cree_english_text_only.csv            # Only Cree-English text
│       └── NKFC_normalized_cree_english.csv      # From 1st trial cleaning
│
├── models/
│   ├── cree_learning_model.pkl                  # Trained text-matching model
│   └── audio/                                   # Whisper+KNN audio matching
│       ├── features.npy / features_large.npy
│       ├── knn_model.pkl / knn_model_large.pkl
│       ├── labels.json / labels_large.json
│       └── paths.json / paths_large.json
│
├── notebooks/
│   ├── text_model.ipynb                          # Text model training notebook
│   ├── speech_model.ipynb                        # Audio model + feature extraction
│   ├── data_gathering.ipynb                      # Initial scraping + formatting
│   ├── data_cleaning_after_1st_training_trial.ipynb  # Analysis of first cleaning pass
│   ├── Cree_text_app.py                          # Streamlit app (text only)
│   ├── cree_audio_matcher_app.py                 # Streamlit app (audio only)
│   ├── cree_learning_model.py                    # Class definition for text model
│   └── cree_learning_app.py                      # Full production app (text + audio)

```

---

## 💡 Features

🧠 Text Learning Module:
- Translate Cree ↔ English using a KNN-based semantic matching model.
- Exercise Mode presents users with a random Cree/English word and asks for its translation.
- Dataset Explorer allows browsing the full dataset.

🔉 Audio Learning Module:
- Match Audio to Word using Whisper for feature extraction and a KNN classifier for nearest match.
- Listen to Dataset mode to play any of the recorded Cree words.

---

## Setup Instructions

1. Clone the repository:
```
git clone https://github.com/Hend-Khaled-Aly/AI-for-Indigenous-Language-Revitalization-in-Canada.git
```

2. Install the required dependencies:
```
- recommended to use virtual environment
pip install -r requirements.txt
```

3. Run the app:
```
streamlit run notebooks/cree_learning_app.py
```

---

## 🔐 Notes

- All models are pre-trained and saved in /models/.
- No API keys required for offline use.
- Audio features were extracted using Whisper (OpenAI) models.
- Text matching uses traditional text embeddings + KNN.
- The full app is in: notebooks/cree_learning_app.py
- Audio files must be placed inside data/audio/ (for mp3 format) and data/wav/ (for wav format).


