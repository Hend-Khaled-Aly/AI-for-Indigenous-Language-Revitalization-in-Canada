import streamlit as st
import torch
import torchaudio
import numpy as np
import json
import joblib
import os
import tempfile
import base64
from transformers import WhisperProcessor, WhisperModel
from sklearn.neighbors import NearestNeighbors
from audiorecorder import audiorecorder

# Set up the Streamlit app page configuration
# - page_title: Sets the title of the browser tab and the app header
# - page_icon: Emoji icon to appear in the browser tab
# - layout: 'wide' makes use of the full width of the browser window for a better UI experience
st.set_page_config(
    page_title="Audio Matching with Whisper + KNN",
    page_icon="🎵",
    layout="wide"
)

# Constants defining paths for the audio matching models and data files:
# - MODELS_DIR: Directory where the audio model files are stored
MODELS_DIR = "../models/audio"    
# - FEATURES_PATH: Numpy file containing precomputed audio feature embeddings
FEATURES_PATH = os.path.join(MODELS_DIR, "features.npy")
# - KNN_MODEL_PATH: Serialized KNN model file for nearest neighbor search
KNN_MODEL_PATH = os.path.join(MODELS_DIR, "knn_model.pkl")
# - LABELS_PATH: JSON file storing labels (Cree words) corresponding to audio samples
LABELS_PATH = os.path.join(MODELS_DIR, "labels.json")
# - PATHS_PATH: JSON file storing the file paths of the audio samples
PATHS_PATH = os.path.join(MODELS_DIR, "paths.json")

@st.cache_resource
def load_whisper_model():
    """
    Load and cache the Whisper speech model along with its processor.
    
    This function:
    - Loads the Whisper processor and model from the pretrained "openai/whisper-large-v3" checkpoint.
    - Sets the model to evaluation mode to disable training-specific behaviors like dropout.
    - Automatically detects if a GPU (CUDA) is available and moves the model to the appropriate device.
    - Caches the loaded model and processor to avoid reloading on every Streamlit rerun, improving performance.
    
    Returns:
        processor: WhisperProcessor object for preprocessing audio inputs.
        model: WhisperModel object for generating embeddings or transcriptions.
        device: String identifier ("cuda" or "cpu") indicating where the model is loaded.
    """
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    model = WhisperModel.from_pretrained("openai/whisper-large-v3")
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return processor, model, device

@st.cache_data
def load_dataset():
    """
    Load precomputed audio features, labels, file paths, and the trained KNN model from disk.
    
    This function:
    - Loads MFCC or Whisper embeddings saved as a NumPy array from FEATURES_PATH.
    - Reads the JSON files containing the corresponding labels (words) and file paths.
    - Loads the serialized KNN model used for nearest neighbor searches.
    - Handles exceptions gracefully by displaying an error message in the Streamlit app.
    - Caches the loaded data to avoid repeated disk I/O on every app rerun, improving performance.
    
    Returns:
        features (np.ndarray): Array of audio feature vectors.
        labels (list of str): List of words corresponding to the features.
        paths (list of str): List of audio file paths corresponding to features.
        knn_model: Trained KNN model for audio similarity queries.
        
    On failure:
        Returns four None values and shows an error message in the app.
    """
    try:
        features = np.load(FEATURES_PATH)
        
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            labels = json.load(f)
        
        with open(PATHS_PATH, "r", encoding="utf-8") as f:
            paths = json.load(f)
        
        knn_model = joblib.load(KNN_MODEL_PATH)
        
        return features, labels, paths, knn_model
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None, None, None, None

def load_audio(path, sample_rate=16000):
    """
    Load and preprocess an audio file to match the expected format for Whisper.

    Args:
        path (str): Path to the input audio file (e.g., WAV file).
        sample_rate (int, optional): Target sampling rate for Whisper. Default is 16,000 Hz.

    Returns:
        tuple:
            - audio (np.ndarray): 1D NumPy array of mono audio samples at the desired sample rate.
            - sample_rate (int): The sample rate used (always 16,000 Hz).

    Process:
        - Loads the audio file using torchaudio.
        - If the file is not already at 16kHz, it resamples it.
        - Converts stereo audio to mono by averaging across channels.
        - Converts the PyTorch tensor to a NumPy array (for compatibility with Whisper processor).
    """
    # Load the audio file; `wav` is a tensor of shape [channels, time], `sr` is the original sample rate
    wav, sr = torchaudio.load(path)
    # If the audio is not at the desired sample rate (16kHz), resample it
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    # Convert to mono by averaging across channels, then convert to NumPy array
    return wav.mean(dim=0).numpy(), sample_rate  # mono

def extract_whisper_embedding(audio_path, processor, model):
    """
    Extract a single Whisper embedding vector from an input audio file.

    Args:
        audio_path (str): Path to the audio file to process (WAV format recommended).
        processor (WhisperProcessor): Pretrained Whisper processor from Hugging Face.
        model (WhisperModel): Pretrained Whisper model used to extract audio embeddings.

    Returns:
        np.ndarray: A 1D NumPy array representing the mean-pooled audio embedding.

    Process:
        - Loads and preprocesses the audio.
        - Uses the Whisper processor to convert raw waveform into model-compatible input.
        - Passes the input through Whisper's encoder to obtain high-level audio features.
        - Applies mean pooling across the time dimension to produce a fixed-length vector.
    """
    audio, sr = load_audio(audio_path)
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    input_features = inputs.input_features.to(model.device)
    
    with torch.no_grad():
        encoder_out = model.encoder(input_features)[0]
    
    return encoder_out.mean(dim=1).cpu().numpy().squeeze()

def query_audio(audio_path, knn_model, labels, paths, processor, model, top_k=3):
    """
    Find and return the top K most similar Cree audio files to a given input audio.

    Args:
        audio_path (str): Path to the input audio file for matching.
        knn_model (NearestNeighbors): Pre-trained k-NN model for audio similarity search.
        labels (List[str]): List of Cree word labels corresponding to the dataset audio files.
        paths (List[str]): List of audio file paths corresponding to the dataset features.
        processor (WhisperProcessor): Whisper processor for converting raw audio to model input.
        model (WhisperModel): Whisper encoder model for generating audio embeddings.
        top_k (int, optional): Number of most similar results to return. Default is 3.

    Returns:
        List[Dict]: A list of dictionaries, each containing:
            - 'rank' (int): Ranking of the match (1 = closest).
            - 'label' (str): Cree word associated with the matched audio.
            - 'distance' (float): Distance between query embedding and matched embedding.
            - 'path' (str): File path of the matched audio file.
    """
    query_emb = extract_whisper_embedding(audio_path, processor, model).reshape(1, -1)
    distances, indices = knn_model.kneighbors(query_emb, n_neighbors=top_k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            'rank': i + 1,
            'label': labels[idx],
            'distance': distances[0][i],
            'path': paths[idx]
        })
    
    return results

def get_audio_player_html(audio_path):
    """
    Generate an HTML audio player to embed audio in a Streamlit app.

    Args:
        audio_path (str): Path to the audio file to be embedded (MP3 or WAV).

    Returns:
        str: HTML string for rendering an audio player in the browser.

    Process:
        - Reads the audio file in binary mode.
        - Encodes the audio content as base64 (required for embedding in HTML).
        - Detects the file format (WAV or MP3) based on the file extension.
        - Constructs and returns an HTML <audio> element with the base64-encoded audio.
        - If any error occurs (e.g., file not found), returns an HTML error message.
    """
    try:
        with open(audio_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
            audio_b64 = base64.b64encode(audio_bytes).decode()
        
        # Determine audio format
        ext = os.path.splitext(audio_path)[1].lower()
        audio_format = "wav" if ext == ".wav" else "mp3"
        
        html = f'''
        <audio controls style="width: 100%;">
            <source src="data:audio/{audio_format};base64,{audio_b64}" type="audio/{audio_format}">
            Your browser does not support the audio element.
        </audio>
        '''
        return html
    except Exception as e:
        return f"<p>Error loading audio: {str(e)}</p>"

def main():
    # Title and description for the app
    st.title("🎵 Audio Matching with Whisper + KNN")
    st.markdown("Upload an audio file to find similar audio clips from the trained dataset.")
    # --------------------------
    # Load models and dataset
    # --------------------------
    with st.spinner("Loading models and dataset..."):
        processor, model, device = load_whisper_model()
        features, labels, paths, knn_model = load_dataset()
    
    if features is None:
        # Stop the app if loading failed
        st.error("Failed to load dataset. Please check if all model files exist in the correct directory.")
        st.stop()
    # Confirmation messages
    st.success(f"✅ Models loaded successfully! Dataset contains {len(labels)} audio samples.")
    st.info(f"🔧 Using device: {device}")
    # --------------------------
    # Sidebar for settings
    # --------------------------
    st.sidebar.header("Settings")
    top_k = st.sidebar.slider("Number of matches to return", min_value=1, max_value=10, value=3)
    
    # Dataset information expandable section
    with st.sidebar.expander("Dataset Info"):
        st.write(f"Total samples: {len(labels)}")
        st.write(f"Feature dimension: {features.shape[1]}")       
        st.write("Sample labels:")
        for i, label in enumerate(labels[:5]):
            st.write(f"• {label}")
        if len(labels) > 5:
            st.write(f"... and {len(labels) - 5} more")
    
    # --------------------------
    # Audio Input Section
    # --------------------------
    st.header("🎙️ Input Audio")
    method = st.radio("Choose input method", ["Upload Audio File", "Record Audio"])

    audio_bytes = None
    audio_filename = None
    # Option 1: Upload audio file
    if method == "Upload Audio File":
        uploaded_file = st.file_uploader("Upload WAV or MP3", type=["wav", "mp3"])
        if uploaded_file:
            audio_bytes = uploaded_file.getvalue()
            audio_filename = uploaded_file.name
            st.audio(audio_bytes, format=f"audio/{audio_filename.split('.')[-1]}")
    # Option 2: Record audio using built-in mic
    elif method == "Record Audio":
        st.info("Click 'Start Recording' then 'Stop Recording'. Wait a moment for preview.")
        recorded_audio = audiorecorder("Start Recording", "Stop Recording")
        if recorded_audio:
            from io import BytesIO
            buffer = BytesIO()
            recorded_audio.export(buffer, format="wav")
            audio_bytes = buffer.getvalue()
            audio_filename = "recorded.wav"
            st.audio(audio_bytes, format="audio/wav")

    # --------------------------
    # Audio Matching Process
    # --------------------------
    if audio_bytes and audio_filename:
        # Save uploaded/recorded audio temporarily for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_filename.split('.')[-1]}") as tmp_file:
            tmp_file.write(audio_bytes)
            temp_path = tmp_file.name

        try:
            st.subheader("🔍 Processing...")
            progress_bar = st.progress(0)
            status = st.empty()
            # Step 1: Extract embedding
            status.text("Extracting embeddings...")
            progress_bar.progress(30)
            # Step 2: Query KNN model for matches
            results = query_audio(temp_path, knn_model, labels, paths, processor, model, top_k)

            progress_bar.progress(100)
            status.text("✅ Done!")
            # --------------------------
            # Display Matching Results
            # --------------------------
            st.subheader("🎯 Top Matches")
            for result in results:
                with st.expander(f"#{result['rank']} - {result['label']} (Distance: {result['distance']:.3f})"):
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.write(f"**Label:** {result['label']}")
                        st.write(f"**Distance:** {result['distance']:.3f}")
                        similarity = 1 - result['distance']
                        st.write(f"**Similarity:** {similarity:.3f}")
                        if similarity > 0.8:
                            st.success("🟢 Very Similar")
                        elif similarity > 0.6:
                            st.warning("🟡 Moderately Similar")
                        else:
                            st.error("🔴 Less Similar")
                    with col2:
                        # Play the matched audio file if it exists
                        if os.path.exists(result['path']):
                            with open(result['path'], 'rb') as f:
                                st.audio(f.read(), format="audio/wav")
                        else:
                            st.error("Audio file not found")
            # --------------------------
            # Summary Metrics
            # --------------------------
            st.subheader("📊 Summary")
            distances = [r["distance"] for r in results]
            col1, col2, col3 = st.columns(3)
            col1.metric("Best Distance", f"{min(distances):.3f}")
            col2.metric("Average", f"{np.mean(distances):.3f}")
            col3.metric("Best Similarity", f"{1 - min(distances):.3f}")

        except Exception as e:
            # Catch and display any runtime errors
            st.error(f"❌ Error: {str(e)}")

        finally:
            # Clean up temporary audio file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    # --------------------------
    # Footer
    # --------------------------
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            <p>Audio Matching System using Whisper + KNN</p>
            <p>Built with Streamlit 🎧</p>
        </div>
        """, unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()