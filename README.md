# 🎧 Music Genre Classification App 🎶  
A Streamlit-based application to classify music genres using deep learning and Mel spectrograms. It supports live audio recording, file upload classification, and batch file segregation by genre.

---

## 💡 Overview
This project uses a trained Convolutional Neural Network (CNN) model to classify audio files into one of the following genres:

`blues`, `classical`, `country`, `disco`, `hiphop`, `jazz`, `metal`, `pop`, `reggae`, `rock`

You can:
- 🎙️ Record live audio and classify it
- 📂 Upload a single audio file for genre prediction
- 🚀 Batch-process and segregate a folder of audio files into genre-wise folders

---

## 🧠 How it Works

1. **Audio Input**: Records audio or loads a file (MP3/WAV)
2. **Preprocessing**: Splits into 4-sec chunks with 2-sec overlap → generates Mel spectrograms
3. **Model Prediction**: Loads the trained `.h5` model and classifies each chunk
4. **Final Result**: Returns the most frequent predicted genre

---

## 📦 Features
- Real-time audio recording via microphone
- Compatible with both `.mp3` and `.wav` formats
- Chunk-based classification for better accuracy
- Beautiful and interactive UI with Streamlit
- Genre-wise file segregation in batch mode

---

## 🛠️ Setup Instructions

1. **Clone the repo**
   ```bash
   [git clone https://github.com/your-username/music-genre-classifier.git](https://github.com/amitg404/Music_genre_classification.git)
   cd music-genre-classifier
