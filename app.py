import os
import shutil
import librosa
import numpy as np
import tensorflow as tf
import streamlit as st
import sounddevice as sd
import wave
from tensorflow import image

# Load the trained model
model_path = r"Trained_model.h5"
model = tf.keras.models.load_model(model_path)

# Define music genres
classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

### Single Audio Preprocessing Function
def load_and_preprocess_data(file_path, target_shape=(150, 150)):
    data = []
    audio_data, sample_rate = librosa.load(file_path, sr=None)

    # Define chunk duration and overlap
    chunk_duration = 4  # seconds
    overlap_duration = 2  # seconds

    # Convert durations to samples
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate

    # Calculate number of chunks
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1

    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        chunk = audio_data[start:end]

        # Compute Mel spectrogram for each chunk
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
        mel_spectrogram = tf.image.resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram)

    return np.array(data)

### Model Prediction for Single Audio File
def model_prediction(X_test):
    y_pred = model.predict(X_test)
    predicted_categories = np.argmax(y_pred, axis=1)
    unique_elements, counts = np.unique(predicted_categories, return_counts=True)
    max_count = np.max(counts)
    max_elements = unique_elements[counts == max_count]
    genre_index = max_elements[0]
    return classes[int(genre_index)]

### Model Prediction with Chunk Analysis
def model_prediction_chunks(X_test):
    y_pred = model.predict(X_test)
    predicted_categories = np.argmax(y_pred, axis=1)
    print("\n--- Predictions for Each Audio Chunk ---\n")
    for idx, category in enumerate(predicted_categories):
        print(f"Chunk {idx + 1}: {classes[category]}")

    # Count class occurrences
    unique_elements, counts = np.unique(predicted_categories, return_counts=True)
    print("\n--- Class Counts for All Chunks ---")
    for element, count in zip(unique_elements, counts):
        print(f"{classes[element]}: {count} chunks")

    # Final overall prediction
    max_count = np.max(counts)
    max_elements = unique_elements[counts == max_count]
    return max_elements[0]

### Audio File Segregation Function
def segregate_music_files(input_folder, output_folder):
    # Create output folder if not exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process files in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.mp3') or file_name.endswith('.wav'):
            file_path = os.path.join(input_folder, file_name)

            X_test = load_and_preprocess_data(file_path)
            genre = model_prediction(X_test)

            # Create genre folder if not exists
            genre_folder = os.path.join(output_folder, genre)
            if not os.path.exists(genre_folder):
                os.makedirs(genre_folder)

            # Copy file to its genre folder
            shutil.copy(file_path, os.path.join(genre_folder, file_name))

### Record Audio Function
def record_audio(output_file, duration, input_device=None):  # Use default input device
    sample_rate = 44100  # Standard sample rate
    channels = 1         # Mono audio for compatibility

    print("Recording audio...")
    try:
        # Record audio
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype='int16', device=input_device)
        sd.wait()  # Block execution until recording finishes

        print("Recording finished.")

        # Save to file
        with wave.open(output_file, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)  # 16-bit audio ('int16')
            wf.setframerate(sample_rate)
            wf.writeframes(recording.tobytes())

        print(f"Audio saved successfully as {output_file}")
    except Exception as e:
        print(f"Error: {e}")
        st.error(f"Recording Error: {e}")

### Streamlit Frontend
def main():
    st.title("Music Genre Classification")

    menu = ["Record Audio and Classify", "Classify a Single File", "Segregate Folder Contents"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Record Audio and Classify":
        st.subheader("Record Audio")
        duration = st.slider("Select recording duration (seconds)", min_value=1, max_value=120, value=5)
        output_file = st.text_input("Enter output file name (e.g., 'recorded_audio.wav')", "recorded_audio.wav")

        # Get available devices
        devices = sd.query_devices()
        input_devices = [f"{i}: {devices[i]['name']}" for i in range(len(devices)) if devices[i]['max_input_channels'] > 0]
        selected_device = st.selectbox("Select Microphone", input_devices)

        # Extract device index
        input_device_index = int(selected_device.split(":")[0]) if selected_device else None

        if st.button("Record and Classify"):
            record_audio(output_file, duration, input_device=input_device_index)
            X_test = load_and_preprocess_data(output_file)
            genre = model_prediction_chunks(X_test)
            st.success(f"Final Prediction: Music Genre --> {classes[genre]}")

    elif choice == "Classify a Single File":
        st.subheader("Classify a Single File")
        audio_file = st.file_uploader("Upload Audio File", type=['mp3', 'wav'])

        if audio_file is not None:
            with open("temp_audio_file.wav", "wb") as f:
                f.write(audio_file.read())
            X_test = load_and_preprocess_data("temp_audio_file.wav")
            genre = model_prediction_chunks(X_test)
            st.success(f"Final Prediction: Music Genre --> {classes[genre]}")

    elif choice == "Segregate Folder Contents":
        st.subheader("Segregate Music Files by Genre")
        input_folder = st.text_input("Enter folder path containing audio files")
        output_folder = st.text_input("Enter folder path for segregated files")

        if st.button("Segregate"):
            segregate_music_files(input_folder, output_folder)
            st.success(f"Files have been segregated into {output_folder}")

if __name__ == "__main__":
    main()