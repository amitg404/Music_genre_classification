
# 🎵 Music Genre Classifier

A Streamlit-based app that classifies audio into genres using a pre-trained deep learning model. It supports audio recording, file uploads, and automatic folder segregation by predicted genre.

## 🚀 How to Run

1. **Clone the Repo**  
   ```bash
   git clone https://github.com/amitg404/Music_genre_classification.git
   cd your-repo-name
   ```

2. **Set Up Virtual Environment**  
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Create Required Folders**  
   Make sure these folders exist in the root directory:
   - `Test_music/` – Drop your audio files here for classification.
   - `segregated_folder/` – Segregated genre-wise files will be saved here.

4. **Download the Pre-trained Model**  
   🔗 [Download `Trained_model.h5`](https://drive.google.com/file/d/1kLvFEu2PBUk5ST13qO7jeBu0Glwv1AEF/view?usp=sharing) and place it in the root project folder.

5. **Run the App**  
   ```bash
   streamlit run app.py
   ```

## 🎯 Features

- 🎙️ Record audio and classify in real-time
- 🎵 Upload single audio files for genre prediction
- 🗂️ Auto-segregate music files into genre-based folders
- 📊 Chunk-wise analysis using Mel Spectrograms

## 🧠 Model Info

The model was trained using Mel-spectrograms extracted from GTZAN music dataset and fine-tuned to identify 10 genres.

## 📦 Requirements

Dependencies listed in `requirements.txt`.

## 💬 Genre Classes

- Blues, Classical, Country, Disco, HipHop, Jazz, Metal, Pop, Reggae, Rock

---

Built with ❤️ to make your music library smarter.
