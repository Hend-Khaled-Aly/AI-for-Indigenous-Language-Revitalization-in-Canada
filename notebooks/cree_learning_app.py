import streamlit as st
import torch
import soundfile as sf
import numpy as np
import json
import joblib
import os
import tempfile
import base64
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from transformers import WhisperProcessor, WhisperModel
from sklearn.neighbors import NearestNeighbors
from cree_learning_model import CreeLearningModel
from scipy.signal import resample
import io
import wave
import streamlit.components.v1 as components
from pathlib import Path
import subprocess
import re
from datetime import datetime
import shutil

WEBRTC_AVAILABLE = False

token = os.getenv("GITHUB_TOKEN")
if not token:
    st.error("GitHub token not found. Please set it in Streamlit secrets.")

# Set the configuration for the Streamlit app page
st.set_page_config(
    page_title="Cree Language Learning App",     # Sets the title shown in the browser tab
    page_icon="🌿",     # Sets the favicon (a small emoji icon representing the app)
    layout="wide"       # Uses the full width of the browser window for better UI spacing
)

# -----------------------------------------------
# Constants for the audio matching component
# These paths are fixed and set relative to the app root
# to ensure compatibility in both local and cloud deployments
# -----------------------------------------------
# Directory containing audio model files and metadata
MODELS_DIR = "models/audio"
# Path to the precomputed Whisper audio feature vectors (NumPy format)
FEATURES_PATH = os.path.join(MODELS_DIR, "features.npy")
# Path to the trained k-NN model used for audio similarity search
KNN_MODEL_PATH = os.path.join(MODELS_DIR, "knn_model.pkl")
# Path to the list of Cree word labels corresponding to the audio samples
LABELS_PATH = os.path.join(MODELS_DIR, "labels.json")
# Path to the list of audio file paths (used for playback and lookup)
PATHS_PATH = os.path.join(MODELS_DIR, "paths.json")

# -----------------------------------------------
# Global variable to store recorded audio
# This is needed so the recorded audio remains available
# even after Streamlit re-runs the script (which happens often)
# -----------------------------------------------
if 'recorded_audio' not in st.session_state:
    st.session_state.recorded_audio = None     # Initialize with None if not already set

# -----------------------------------------------
# AudioRecorder class
# A custom utility to handle in-app audio recording,
# frame processing, saving to WAV, and clearing data.
# -----------------------------------------------
class AudioRecorder:
    def __init__(self):
        # Initialize an empty list to store audio frames as they come in
        self.frames = []
        # Set sample rate to 16kHz, which is required for compatibility with Whisper
        self.sample_rate = 16000
        
    def process_audio_frame(self, frame):
        """
        Process a single audio frame.
        Args:
            frame: An audio frame from the Streamlit recorder.
        Returns:
            The original frame (unmodified), but stores processed samples internally.
        """
        # Convert frame to a NumPy array of type float32
        sound = frame.to_ndarray().astype(np.float32)
        # If stereo (2 channels), average them to convert to mono
        if len(sound.shape) > 1:
            sound = sound.mean(axis=1)
        # Append this frame’s samples to the list of recorded frames
        self.frames.extend(sound)
        return frame
    
    def get_audio_data(self):
        """
        Get the entire recorded audio as a NumPy array.
        Returns:
            np.ndarray or None: All recorded samples as a 1D float array.
        """
        if self.frames:
            return np.array(self.frames, dtype=np.float32)
        return None
    
    def save_as_wav(self, filename):
        """
        Save the recorded audio to a WAV file on disk.
        Args:
            filename (str): Path to the output WAV file.
        Returns:
            bool: True if saving was successful, False if there was no data to save.
        """
        if self.frames:
            audio_data = np.array(self.frames, dtype=np.float32)
            # Normalize audio
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            sf.write(filename, audio_data, self.sample_rate)
            return True
        return False
    
    def clear(self):
        """
        Clear all previously recorded frames.
        Useful if the user wants to re-record.
        """
        self.frames = []

def check_audio_files():
    """
    Check whether all required audio model files are available.

    Returns:
        tuple:
            - (bool): True if all files exist, False otherwise.
            - (list of str): A list of missing file paths, if any.

    Required files:
        - features.npy: Precomputed Whisper embeddings.
        - knn_model.pkl: Trained k-NN model.
        - labels.json: Cree word labels for each audio file.
        - paths.json: File paths of the stored dataset audio clips.
    """
    required_files = [FEATURES_PATH, KNN_MODEL_PATH, LABELS_PATH, PATHS_PATH]
    missing_files = []
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    return len(missing_files) == 0, missing_files


@st.cache_resource
def load_whisper_model():
    """
    Load and cache the Whisper processor and model.

    Returns:
        tuple:
            - processor: WhisperProcessor object for preprocessing audio.
            - model: WhisperModel for generating audio embeddings.
            - device: Device identifier ("cpu" in this case).
        On failure:
            Returns (None, None, None) and shows an error message in the app.

    Notes:
        - Uses the small Whisper model to reduce memory usage.
        - Forces CPU execution for cloud compatibility (no GPU dependencies).
        - Caches result using Streamlit’s `@st.cache_resource` to improve performance.
    """
    try:
        # Use smaller model for cloud deployment to reduce memory usage
        model_name = "openai/whisper-small" 
        # model_name = "openai/whisper-large-v3"
        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperModel.from_pretrained(model_name)
        model.eval()
        
        # Force CPU usage on cloud to avoid CUDA issues
        device = "cpu"  # Force CPU for cloud deployment
        model = model.to(device)
        
        return processor, model, device
    except Exception as e:
        st.error(f"Failed to load Whisper model: {str(e)}")
        return None, None, None

@st.cache_data
def load_dataset():
    """
    Load the precomputed audio features, labels, paths, and k-NN model.

    Returns:
        tuple:
            - features (np.ndarray): Audio feature vectors (Whisper embeddings).
            - labels (List[str]): Cree word labels for each audio file.
            - paths (List[str]): File paths to corresponding audio files.
            - knn_model: Trained k-NN model for similarity search.

        On failure:
            Returns (None, None, None, None) and logs helpful debugging info.

    Notes:
        - Uses @st.cache_data to avoid reloading on every Streamlit rerun.
        - Includes robust error handling and debugging support.
    """
    try:
        # Check if files exist first
        files_exist, missing_files = check_audio_files()
        if not files_exist:
            st.error(f"Missing required files: {missing_files}")
            return None, None, None, None
        
        features = np.load(FEATURES_PATH)
        
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            labels = json.load(f)
        
        with open(PATHS_PATH, "r", encoding="utf-8") as f:
            paths = json.load(f)
        
        knn_model = joblib.load(KNN_MODEL_PATH)
        
        return features, labels, paths, knn_model
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        st.error(f"Current working directory: {os.getcwd()}")
        st.error(f"Files in current directory: {os.listdir('.')}")
        return None, None, None, None

# @st.cache_resource
def load_cree_model():
    """
    Load and cache the Cree text matching model.

    Returns:
        CreeLearningModel: The trained model instance used in the text app.
        On failure: Returns None and logs helpful debugging messages.

    Notes:
        - Uses a fixed file path compatible with cloud deployment.
        - Includes robust error handling for missing files.
        - Uses @st.cache_resource to avoid reloading the model on each rerun.
    """
    try:
        # Fixed path for cloud deployment
        model_path = "models/cree_learning_model.pkl" 
        
        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found at: {model_path}")
            st.error(f"Current directory: {os.getcwd()}")
            st.error(f"Available files: {os.listdir('.')}")
            if os.path.exists('models'):
                st.error(f"Files in models/: {os.listdir('models')}")
            return None
        
        model = CreeLearningModel()
        model.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading Cree model: {str(e)}")
        return None


def load_audio(path, sample_rate=16000):
    """
    Load and preprocess an audio file using `soundfile`.

    Args:
        path (str): Path to the audio file (WAV format).
        sample_rate (int): Desired sampling rate for the audio (default is 16000 for Whisper).

    Returns:
        tuple:
            - audio (np.ndarray): 1D NumPy array of audio samples (mono).
            - sample_rate (int): Final sample rate after resampling.

        On failure:
            Returns (None, None) and logs the error.

    Notes:
        - Uses the `soundfile` (sf) library instead of torchaudio for wider compatibility.
        - Converts stereo to mono by averaging channels.
        - Resamples audio to 16kHz if necessary for Whisper compatibility.
    """
    try:
        # Use soundfile to load WAV
        audio, sr = sf.read(path)

        # Ensure mono (average across channels if stereo)
        if len(audio.shape) == 2:
            audio = audio.mean(axis=1)

        # Resample if needed
        if sr != sample_rate:
            num_samples = int(len(audio) * float(sample_rate) / sr)
            audio = resample(audio, num_samples)

        return audio, sample_rate
    except Exception as e:
        st.error(f"Error loading audio file: {str(e)}")
        return None, None
        

def extract_whisper_embedding(audio_path, processor, model):
    """
    Extract Whisper encoder embeddings from an audio file.

    Args:
        audio_path (str): Path to the input audio file.
        processor (WhisperProcessor): Preprocessing tool for audio inputs.
        model (WhisperModel): Pretrained Whisper model (encoder used for feature extraction).

    Returns:
        np.ndarray or None:
            - A 1D NumPy array representing the mean Whisper encoder embedding of the audio.
            - Returns None if loading or processing fails.

    Notes:
        - Converts raw audio into model-ready input using the processor.
        - Uses the encoder of the Whisper model to extract deep audio representations.
        - Aggregates embeddings by averaging across time steps.
        - Includes error handling to avoid crashing the app if audio is bad or missing.
    """
    try:
        audio, sr = load_audio(audio_path)
        if audio is None:
            return None
        
        inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
        input_features = inputs.input_features.to(model.device)
        
        with torch.no_grad():
            encoder_out = model.encoder(input_features)[0]
        
        return encoder_out.mean(dim=1).cpu().numpy().squeeze()
    except Exception as e:
        st.error(f"Error extracting embeddings: {str(e)}")
        return None

def extract_whisper_embedding_from_array(audio_array, sample_rate, processor, model):
    """
    Extract Whisper encoder embeddings from a raw audio NumPy array.

    Args:
        audio_array (np.ndarray): 1D float array containing audio samples (must be mono).
        sample_rate (int): Sampling rate of the audio data (should be 16000 for Whisper).
        processor (WhisperProcessor): Used to preprocess audio into model input format.
        model (WhisperModel): Whisper model to extract embeddings from the encoder.

    Returns:
        np.ndarray or None:
            - A 1D NumPy array representing the averaged Whisper encoder embedding.
            - Returns None if an error occurs.

    Notes:
        - This function is used when audio is already in memory (not from file).
        - Similar to extract_whisper_embedding(), but input is an array not a file path.
        - Embedding is computed by averaging encoder outputs across time steps.
        - Includes error handling to keep the app running on failure.
    """
    try:
        inputs = processor(audio_array, sampling_rate=sample_rate, return_tensors="pt")
        input_features = inputs.input_features.to(model.device)
        
        with torch.no_grad():
            encoder_out = model.encoder(input_features)[0]
        
        return encoder_out.mean(dim=1).cpu().numpy().squeeze()
    except Exception as e:
        st.error(f"Error extracting embeddings: {str(e)}")
        return None

def query_audio(audio_path, knn_model, labels, paths, processor, model, top_k=3):
    """
    Perform audio similarity search using Whisper embeddings and a k-NN model.

    Args:
        audio_path (str): Path to the input audio file to be matched.
        knn_model (NearestNeighbors): Pre-trained k-NN model for finding nearest embeddings.
        labels (list of str): List of Cree words corresponding to dataset audio samples.
        paths (list of str): List of file paths for the dataset audio samples.
        processor (WhisperProcessor): Preprocessing tool for audio input.
        model (WhisperModel): Whisper model used to extract embeddings.
        top_k (int): Number of nearest matches to return (default is 3).

    Returns:
        list of dict: Each dict contains:
            - rank (int): Rank of the result (1 is most similar).
            - label (str): Cree word label for the matched sample.
            - distance (float): Euclidean distance between query and match.
            - path (str): File path of the matched audio sample.

        Returns an empty list if an error occurs or embedding extraction fails.

    Notes:
        - Uses Whisper to extract audio embedding from input file.
        - Finds the top_k closest matches using Euclidean distance.
        - Returns human-readable results with ranks and metadata.
    """
    try:
        query_emb = extract_whisper_embedding(audio_path, processor, model)
        if query_emb is None:
            return []
        
        query_emb = query_emb.reshape(1, -1)
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
    except Exception as e:
        st.error(f"Error querying audio: {str(e)}")
        return []

def query_audio_from_array(audio_array, sample_rate, knn_model, labels, paths, processor, model, top_k=3):
    """
    Perform audio similarity search using a NumPy audio array (in-memory audio).

    Args:
        audio_array (np.ndarray): 1D float array of audio samples (mono).
        sample_rate (int): Sampling rate of the input audio.
        knn_model (NearestNeighbors): Trained k-NN model for nearest neighbor search.
        labels (list of str): Cree word labels for dataset audio files.
        paths (list of str): File paths to dataset audio files.
        processor (WhisperProcessor): Whisper processor for audio preprocessing.
        model (WhisperModel): Whisper model to extract embeddings.
        top_k (int): Number of top matches to return (default is 3).

    Returns:
        list of dict: Each result contains:
            - rank (int): Ranking (1 = closest match).
            - label (str): Cree word label of the matched audio.
            - distance (float): Distance score (lower = more similar).
            - path (str): File path of the matched sample.

        Returns an empty list if extraction fails or an error occurs.

    Notes:
        - This version takes raw audio data in memory (not from disk).
        - Embedding is computed using the Whisper encoder, then matched using k-NN.
        - Useful when recording audio live in the app (no file saved).
    """
    try:
        query_emb = extract_whisper_embedding_from_array(audio_array, sample_rate, processor, model)
        if query_emb is None:
            return []
        
        query_emb = query_emb.reshape(1, -1)
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
    except Exception as e:
        st.error(f"Error querying audio: {str(e)}")
        return []

def get_audio_player_html(audio_path):
    """
    Generate HTML for an inline audio player that can be embedded in Streamlit.

    Args:
        audio_path (str): Path to the audio file to be played (WAV or MP3).

    Returns:
        str: A string containing HTML markup for an audio player.
             Returns an error message in HTML format if the file is missing or cannot be read.

    Notes:
        - The audio file is read and encoded in base64 so it can be embedded directly in HTML.
        - Automatically detects audio format (WAV or MP3) based on file extension.
        - Includes error handling for file access issues.
    """
    try:
        if not os.path.exists(audio_path):
            return "<p>Audio file not found</p>"
        
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


def simple_audio_recorder():
    """
    Display an embedded browser-based audio recorder in Streamlit using HTML + JavaScript.

    Functionality:
    - Allows the user to record audio directly in the browser.
    - Converts the audio to WAV format (16kHz, mono) for Whisper compatibility.
    - Provides download link for the recorded file.
    - No audio data is sent to the server — all processing is done client-side.

    UI Elements:
    - Start and Stop recording buttons.
    - Embedded audio playback for preview.
    - Download link for the recorded WAV file.

    Notes:
    - Uses Streamlit's components.html to embed raw HTML and JS.
    - Uses the Web Audio API for recording and exporting audio.
    """

    st.subheader("🎙️ Record Audio and Download WAV")

    components.html("""
    <style>
        .recorder-container {
            padding: 20px;
            border: 2px dashed #ccc;
            border-radius: 12px;
            background-color: #f9f9f9;
            text-align: center;
        }
        .recorder-button {
            font-size: 16px;
            padding: 10px 20px;
            margin: 5px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
        }
        .start-btn { background-color: #f44336; color: white; }
        .stop-btn { background-color: #2196F3; color: white; }
        .download-link {
            display: inline-block;
            margin-top: 15px;
            font-weight: bold;
            color: #4CAF50;
        }
    </style>

    <div class="recorder-container">
        <button class="recorder-button start-btn" onclick="startRecording()">🔴 Start Recording</button>
        <button class="recorder-button stop-btn" onclick="stopRecording()" id="stopBtn" disabled>⏹️ Stop</button>
        <p id="statusText">Click start to begin recording...</p>
        <audio id="audioPlayback" controls style="display:none; margin-top: 15px;"></audio>
        <br/>
        <a id="downloadLink" class="download-link" style="display:none;" download="recording.wav">💾 Download Recording</a>
    </div>

    <script>
        let audioContext;
        let mediaStream;
        let processor;
        let source;
        let audioData = [];

        function startRecording() {
            document.getElementById("statusText").innerText = "🎙️ Recording...";
            document.getElementById("stopBtn").disabled = false;
            audioData = [];

            navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
                audioContext = new AudioContext({ sampleRate: 16000 });
                mediaStream = stream;
                source = audioContext.createMediaStreamSource(stream);
                processor = audioContext.createScriptProcessor(4096, 1, 1);

                source.connect(processor);
                processor.connect(audioContext.destination);

                processor.onaudioprocess = e => {
                    const input = e.inputBuffer.getChannelData(0);
                    audioData.push(new Float32Array(input));
                };
            });
        }

        function stopRecording() {
            document.getElementById("stopBtn").disabled = true;
            processor.disconnect();
            source.disconnect();
            mediaStream.getTracks().forEach(track => track.stop());

            const flatData = flattenArray(audioData);
            const wavBlob = encodeWAV(flatData, 16000);
            const url = URL.createObjectURL(wavBlob);

            const audioEl = document.getElementById("audioPlayback");
            audioEl.src = url;
            audioEl.style.display = "block";
            audioEl.load();

            const link = document.getElementById("downloadLink");
            link.href = url;
            link.style.display = "inline-block";

            document.getElementById("statusText").innerText = "✅ Recording complete!";
        }

        function flattenArray(chunks) {
            let length = chunks.reduce((sum, arr) => sum + arr.length, 0);
            let result = new Float32Array(length);
            let offset = 0;
            for (let chunk of chunks) {
                result.set(chunk, offset);
                offset += chunk.length;
            }
            return result;
        }

        function encodeWAV(samples, sampleRate) {
            const buffer = new ArrayBuffer(44 + samples.length * 2);
            const view = new DataView(buffer);

            function writeString(view, offset, string) {
                for (let i = 0; i < string.length; i++) {
                    view.setUint8(offset + i, string.charCodeAt(i));
                }
            }

            function floatTo16BitPCM(output, offset, input) {
                for (let i = 0; i < input.length; i++, offset += 2) {
                    const s = Math.max(-1, Math.min(1, input[i]));
                    output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
                }
            }

            writeString(view, 0, "RIFF");
            view.setUint32(4, 36 + samples.length * 2, true);
            writeString(view, 8, "WAVE");
            writeString(view, 12, "fmt ");
            view.setUint32(16, 16, true);
            view.setUint16(20, 1, true);
            view.setUint16(22, 1, true);
            view.setUint32(24, sampleRate, true);
            view.setUint32(28, sampleRate * 2, true);
            view.setUint16(32, 2, true);
            view.setUint16(34, 16, true);
            writeString(view, 36, "data");
            view.setUint32(40, samples.length * 2, true);

            floatTo16BitPCM(view, 44, samples);

            return new Blob([view], { type: "audio/wav" });
        }
    </script>
    """, height=380)

    st.markdown("""
    ### 📋 Instructions
    1. Click **Start Recording** and allow microphone access
    2. Speak into your microphone
    3. Click **Stop** when done
    4. Play back or **Download WAV**
    5. Upload the WAV file in the next section
    """)

    st.info("💡 This tool runs fully in-browser. No server-side audio is saved or uploaded.")


def audio_listening_page():
    """
    Streamlit page for listening to all available Cree audio files in the dataset.

    Features:
    - Displays the number of available audio samples (from the /data/wav directory).
    - Allows users to search for words using a text input.
    - Displays each word (derived from filename) alongside a button to play the corresponding audio.

    Notes:
    - Assumes WAV files are stored in `../data/wav/` relative to this script.
    - All audio files must be in .wav format.
    """
    st.header("🎧 Listen to Cree Audio Dataset")

    AUDIO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "wav"))

    if not os.path.exists(AUDIO_DIR):
        st.error(f"Audio folder not found at: {AUDIO_DIR}")
        return

    files = [f for f in os.listdir(AUDIO_DIR) if f.endswith((".wav"))]
    total_files = len(files)

    if total_files == 0:
        st.info("No audio files found in the dataset.")
        return

    # Optional search bar
    search_query = st.text_input("🔍 Search Words")
    if search_query:
        files = [f for f in files if search_query.lower() in f.lower()]

    st.markdown(f"### 🎵 {total_files} audio files found")

    # Sort and display each word with a speaker icon
    for fname in sorted(files):
        # Convert filename to word (remove extension, replace underscores)
        word = Path(fname).stem.replace("_", " ")
        filepath = os.path.join(AUDIO_DIR, fname)

        with open(filepath, "rb") as f:
            audio_bytes = f.read()

        cols = st.columns([3, 1])
        with cols[0]:
            st.markdown(f"**{word}**")

        with cols[1]:
            if st.button("🔊", key=f"play_{fname}"):
                st.audio(audio_bytes, format="audio/wav" if fname.endswith("wav") else "audio/mp3")

def sanitize_filename(word):
    """
    Convert a Cree word to a valid filename.
    
    Args:
        word (str): The Cree word to convert
        
    Returns:
        str: Sanitized filename without extension
    """
    # Remove extra spaces and convert to lowercase
    sanitized = word.strip().lower()
    # Replace spaces with underscores
    sanitized = sanitized.replace(" ", "_")
    # Remove any special characters except underscores and hyphens
    sanitized = re.sub(r'[^\w\-_]', '', sanitized)
    # Remove multiple underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    
    return sanitized

def save_audio_to_dataset(audio_bytes, word, audio_dir="../data/wav"):
    """
    Save uploaded audio file to the dataset directory.
    
    Args:
        audio_bytes (bytes): Raw audio file bytes
        word (str): Cree word that this audio represents
        audio_dir (str): Directory to save the audio file
        
    Returns:
        tuple: (success: bool, filepath: str, message: str)
    """
    try:
        # Create directory if it doesn't exist
        audio_dir = os.path.abspath(audio_dir)
        os.makedirs(audio_dir, exist_ok=True)
        
        # Sanitize the word for filename
        filename = sanitize_filename(word)
        if not filename:
            return False, "", "Invalid word - cannot create filename"
        
        # Create full filepath
        filepath = os.path.join(audio_dir, f"{filename}.wav")
        
        # Check if file already exists
        if os.path.exists(filepath):
            return False, filepath, f"Audio file already exists for word: {word}"
        
        # Save the audio file
        with open(filepath, "wb") as f:
            f.write(audio_bytes)
        
        return True, filepath, f"Successfully saved audio for: {word}"
        
    except Exception as e:
        return False, "", f"Error saving audio: {str(e)}"

def update_audio_models(new_audio_path, new_label, processor, model, 
                       features_path=None, knn_model_path=None, 
                       labels_path=None, paths_path=None):
    """
    Update the audio models with a new audio file.
    
    Args:
        new_audio_path (str): Path to the new audio file
        new_label (str): Label for the new audio
        processor: Whisper processor
        model: Whisper model
        features_path (str): Path to features.npy file
        knn_model_path (str): Path to knn_model.pkl file
        labels_path (str): Path to labels.json file
        paths_path (str): Path to paths.json file
        
    Returns:
        tuple: (success: bool, message: str)
    """
    try:
        # Set default paths if not provided
        if features_path is None:
            features_path = os.path.join("models/audio", "features.npy")
        if knn_model_path is None:
            knn_model_path = os.path.join("models/audio", "knn_model.pkl")
        if labels_path is None:
            labels_path = os.path.join("models/audio", "labels.json")
        if paths_path is None:
            paths_path = os.path.join("models/audio", "paths.json")
        
        # Extract embedding for the new audio
        new_embedding = extract_whisper_embedding(new_audio_path, processor, model)
        if new_embedding is None:
            return False, "Failed to extract embedding from new audio"
        
        # Load existing data
        try:
            existing_features = np.load(features_path)
            with open(labels_path, 'r', encoding='utf-8') as f:
                existing_labels = json.load(f)
            with open(paths_path, 'r', encoding='utf-8') as f:
                existing_paths = json.load(f)
        except FileNotFoundError:
            # If files don't exist, create new ones
            existing_features = np.empty((0, new_embedding.shape[0]))
            existing_labels = []
            existing_paths = []
        
        # Check if label already exists
        if new_label in existing_labels:
            return False, f"Label '{new_label}' already exists in the dataset"
        
        # Add new data
        updated_features = np.vstack([existing_features, new_embedding.reshape(1, -1)])
        updated_labels = existing_labels + [new_label]
        updated_paths = existing_paths + [new_audio_path]
        
        # Create and train new k-NN model
        from sklearn.neighbors import NearestNeighbors
        knn_model = NearestNeighbors(n_neighbors=min(5, len(updated_labels)), 
                                   metric='euclidean', 
                                   algorithm='auto')
        knn_model.fit(updated_features)
        
        # Create backup of existing files
        backup_suffix = datetime.now().strftime("_%Y%m%d_%H%M%S")
        for path in [features_path, knn_model_path, labels_path, paths_path]:
            if os.path.exists(path):
                backup_path = path + backup_suffix + ".bak"
                shutil.copy2(path, backup_path)
        
        # Save updated files
        np.save(features_path, updated_features)
        joblib.dump(knn_model, knn_model_path)
        
        with open(labels_path, 'w', encoding='utf-8') as f:
            json.dump(updated_labels, f, ensure_ascii=False, indent=2)
        
        with open(paths_path, 'w', encoding='utf-8') as f:
            json.dump(updated_paths, f, ensure_ascii=False, indent=2)
        
        return True, f"Successfully updated models with new audio for: {new_label}"
        
    except Exception as e:
        return False, f"Error updating models: {str(e)}"

def push_audio_to_github(audio_filepath, word):
    """
    Push a new audio file to GitHub repository.
    
    Args:
        audio_filepath (str): Local path to the audio file
        word (str): The Cree word for this audio
        
    Returns:
        tuple: (success: bool, message: str)
    """
    try:
        from github import InputGitAuthor, Github
        
        token = st.secrets["GITHUB_TOKEN"]
        repo_url = st.secrets["GITHUB_REPO_URL"] 
        author_name = st.secrets["GIT_AUTHOR_NAME"]
        author_email = st.secrets["GIT_AUTHOR_EMAIL"]

        # Extract repo name from URL
        repo_name = repo_url.split("/")[-1].replace(".git", "")
        username = repo_url.split("/")[-2]

        g = Github(token)
        repo = g.get_repo(f"{username}/{repo_name}")
        author = InputGitAuthor(author_name, author_email)

        # Read the audio file
        with open(audio_filepath, "rb") as f:
            audio_content = f.read()

        # Create the GitHub path
        filename = os.path.basename(audio_filepath)
        github_path = f"data/wav/{filename}"

        try:
            # Check if file already exists
            existing_file = repo.get_contents(github_path)
            return False, f"Audio file already exists in repository: {filename}"
        except:
            # File doesn't exist, create it
            repo.create_file(
                path=github_path,
                message=f"📢 Add new Cree audio: {word} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                content=audio_content,
                author=author,
                branch="main"
            )
            return True, f"Successfully uploaded audio for '{word}' to GitHub"

    except Exception as e:
        return False, f"Error pushing to GitHub: {str(e)}"

def push_updated_audio_models_to_github():
    """
    Push updated audio model files to GitHub.
    
    Returns:
        tuple: (success: bool, message: str)
    """
    try:
        from github import InputGitAuthor, Github
        
        token = st.secrets["GITHUB_TOKEN"]
        repo_url = st.secrets["GITHUB_REPO_URL"] 
        author_name = st.secrets["GIT_AUTHOR_NAME"]
        author_email = st.secrets["GIT_AUTHOR_EMAIL"]

        # Extract repo name from URL
        repo_name = repo_url.split("/")[-1].replace(".git", "")
        username = repo_url.split("/")[-2]

        g = Github(token)
        repo = g.get_repo(f"{username}/{repo_name}")
        author = InputGitAuthor(author_name, author_email)

        # Files to push
        model_files = {
            "models/audio/features.npy": "binary",
            "models/audio/knn_model.pkl": "binary",
            "models/audio/labels.json": "text",
            "models/audio/paths.json": "text"
        }

        success_count = 0
        for file_path, file_type in model_files.items():
            if not os.path.exists(file_path):
                continue
                
            with open(file_path, "rb") as f:
                content_bytes = f.read()

            if file_type == "text":
                encoded_content = content_bytes.decode("utf-8")
            else:
                encoded_content = content_bytes

            try:
                remote_file = repo.get_contents(file_path)
                # Update existing file
                repo.update_file(
                    path=file_path,
                    message=f"🔄 Update audio model - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    content=encoded_content,
                    sha=remote_file.sha,
                    author=author,
                    branch="main"
                )
                success_count += 1
            except:
                # Create new file
                repo.create_file(
                    path=file_path,
                    message=f"📦 Create audio model - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    content=encoded_content,
                    author=author,
                    branch="main"
                )
                success_count += 1

        return True, f"Successfully updated {success_count} audio model files on GitHub"

    except Exception as e:
        return False, f"Error pushing audio models to GitHub: {str(e)}"

def add_audio_to_dataset_ui():
    """
    Streamlit UI for adding new audio to the dataset.
    """
    st.subheader("➕ Add New Audio to Dataset")
    
    with st.form("add_audio_form"):
        st.markdown("### 📤 Upload New Audio")
        
        # Word input
        word = st.text_input(
            "Cree Word *", 
            placeholder="Enter the Cree word (e.g., 'acim' or 'acosis awa')",
            help="Spaces will be converted to underscores in the filename"
        )
        
        # Audio file upload
        uploaded_audio = st.file_uploader(
            "Audio File *", 
            type=["wav"],
            help="Upload a WAV file (recommended: 16kHz, mono, 2-30 seconds)"
        )
        
        # Preview section
        if word:
            filename = sanitize_filename(word)
            if filename:
                st.info(f"💾 This will be saved as: `{filename}.wav`")
            else:
                st.error("❌ Invalid word - cannot create filename")
        
        if uploaded_audio:
            st.audio(uploaded_audio.getvalue(), format="audio/wav")
        
        # Submit button
        submitted = st.form_submit_button("🔥 Add to Dataset", type="primary")
        
        if submitted:
            if not word or not uploaded_audio:
                st.error("❌ Please provide both a word and an audio file")
                return
            
            if not sanitize_filename(word):
                st.error("❌ Invalid word - cannot create filename")
                return
            
            # Save audio file
            with st.spinner("💾 Saving audio file..."):
                success, filepath, message = save_audio_to_dataset(
                    uploaded_audio.getvalue(), 
                    word
                )
            
            if not success:
                st.error(f"❌ {message}")
                return
            
            st.success(f"✅ {message}")
            
            # Update models
            with st.spinner("🔄 Updating audio models..."):
                processor, model, device = load_whisper_model()
                if processor and model:
                    success, message = update_audio_models(
                        filepath, word, processor, model
                    )
                    
                    if success:
                        st.success(f"✅ {message}")
                        
                        # Ask if user wants to push to GitHub
                        if st.button("🚀 Push to GitHub Repository"):
                            with st.spinner("📤 Uploading to GitHub..."):
                                # Push audio file
                                audio_success, audio_msg = push_audio_to_github(filepath, word)
                                if audio_success:
                                    st.success(f"✅ {audio_msg}")
                                else:
                                    st.error(f"❌ {audio_msg}")
                                
                                # Push model files
                                model_success, model_msg = push_updated_audio_models_to_github()
                                if model_success:
                                    st.success(f"✅ {model_msg}")
                                else:
                                    st.error(f"❌ {model_msg}")
                    else:
                        st.error(f"❌ {message}")
                else:
                    st.error("❌ Failed to load Whisper models")
    
    # Display current dataset stats
    with st.expander("📊 Current Dataset Statistics"):
        try:
            labels_path = os.path.join("models/audio", "labels.json")
            if os.path.exists(labels_path):
                with open(labels_path, 'r', encoding='utf-8') as f:
                    labels = json.load(f)
                st.write(f"Total audio samples: {len(labels)}")
                st.write("Recent additions:")
                for label in labels[-5:]:
                    st.write(f"• {label}")
            else:
                st.write("No audio dataset found")
        except Exception as e:
            st.error(f"Error reading dataset: {str(e)}")

def audio_learning_app():
    """Main function to run the Audio Learning/Matching App with two modes:
    1. Listening to the dataset
    2. Matching uploaded/recorded audio using Whisper + KNN
    """
    # Sidebar for audio page selection
    with st.sidebar:
        st.header("Audio Options")
        selected_audio_page = st.radio("Choose a view:", ["🎧 Listen to Dataset", "🎙️ Match Audio", "➕ Add Audio"])

    # Render based on user selection
    if selected_audio_page == "🎧 Listen to Dataset":
        audio_listening_page()
        return  # Exit to avoid loading models, etc.
    elif selected_audio_page == "➕ Add Audio":
        add_audio_to_dataset_ui()
        return

    st.header("🎵 Audio Matching")
    st.markdown("Upload an audio file or record audio to find similar audio clips from the trained dataset.")
    
    # Check if required files exist first
    files_exist, missing_files = check_audio_files()
    if not files_exist:
        st.error("❌ Audio feature files are missing!")
        st.error("Required files for audio matching:")
        for file in missing_files:
            st.error(f"• {file}")
        st.info("Please ensure all model files are uploaded to your Streamlit Cloud repository in the correct directory structure.")
        return
    
    # Load models and dataset
    with st.spinner("Loading models and dataset..."):
        processor, model, device = load_whisper_model()
        if processor is None or model is None:
            st.error("Failed to load Whisper model. Please check your internet connection and try again.")
            return
        
        features, labels, paths, knn_model = load_dataset()
    
    if features is None:
        st.error("Failed to load dataset. Please check if all model files exist in the correct directory.")
        return
    
    st.success(f"✅ Models loaded successfully! Dataset contains {len(labels)} audio samples.")
    st.info(f"🔧 Using device: {device}")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("Audio Settings")
        top_k = st.slider("Number of matches to return", min_value=1, max_value=10, value=3)
        
        # Display dataset info
        with st.expander("Dataset Info"):
            st.write(f"Total samples: {len(labels)}")
            st.write(f"Feature dimension: {features.shape[1]}")       
            st.write("Sample labels:")
            for i, label in enumerate(labels[:5]):
                st.write(f"• {label}")
            if len(labels) > 5:
                st.write(f"... and {len(labels) - 5} more")
    
    # Audio Input Section
    st.subheader("🎙️ Input Audio")
    
    # Tabs for different input methods
    tab1, tab2 = st.tabs(["📁 Upload File", "🎙️ Record Audio"])
    
    audio_source = None
    use_recorded = False
    
    with tab1:
        st.markdown("**📏 Recommended:** 2-30 seconds, 16kHz sample rate")
        uploaded_file = st.file_uploader("Upload Audio File", type=["wav", "mp3"])
        
        if uploaded_file:
            audio_bytes = uploaded_file.getvalue()
            audio_filename = uploaded_file.name
            file_extension = audio_filename.split('.')[-1].lower()
            
            # Show file info
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"📁 **File**: {audio_filename}")
                st.write(f"📏 **Size**: {len(audio_bytes):,} bytes")
            with col2:
                st.write(f"🎵 **Format**: {file_extension.upper()}")
                if file_extension == 'mp3':
                    st.warning("⚠️ MP3 format may require conversion")
            
            st.audio(audio_bytes, format=f"audio/{file_extension}")
            audio_source = uploaded_file
    
    with tab2:
        # Use simple HTML5 recorder
        simple_audio_recorder()
        st.info("🎙️ Use the recorder above to record audio, then upload the downloaded file in the 'Upload File' tab.")
    
    # Process audio if available
    if audio_source:
        if st.button("🔍 Find Similar Audio", type="primary"):
            temp_path = None
            
            try:
                if use_recorded:
                    # Use recorded audio directly
                    temp_path = audio_source
                else:
                    # Handle uploaded file
                    audio_bytes = audio_source.getvalue()
                    audio_filename = audio_source.name
                    file_extension = audio_filename.split('.')[-1].lower()
                    
                    conversion_needed = file_extension == 'mp3'
                    
                    if conversion_needed:
                        st.info("🔄 MP3 detected, converting to WAV...")
                        st.error("❌ MP3 conversion failed.")
                        st.markdown("""
                        **🛠️ Manual Conversion Required:**
                            
                        1. **Online Converters**: Use [CloudConvert](https://cloudconvert.com/mp3-to-wav) or [Online-Convert](https://audio.online-convert.com/convert-to-wav)
                        2. **Desktop Software**: Use Audacity (free) or other audio editors
                        3. **Settings**: Convert to WAV, 16kHz sample rate, mono channel
                            
                        **Or try this quick fix:**
                        - Record your audio again and save as WAV format
                        """)
                        return
                    else:
                        # Create temporary file for WAV
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
                            tmp_file.write(audio_bytes)
                            temp_path = tmp_file.name

                if temp_path and os.path.exists(temp_path):
                    st.subheader("🔍 Processing...")
                    progress_bar = st.progress(0)
                    status = st.empty()

                    status.text("Extracting embeddings...")
                    progress_bar.progress(30)

                    if use_recorded:
                        # Process recorded audio
                        audio_data = st.session_state.audio_recorder.get_audio_data()
                        results = query_audio_from_array(audio_data, 16000, knn_model, labels, paths, processor, model, top_k)
                    else:
                        # Process uploaded audio
                        results = query_audio(temp_path, knn_model, labels, paths, processor, model, top_k)

                    if not results:
                        st.error("Failed to process audio.")
                        st.info("**Possible solutions:**")
                        st.info("• Try converting MP3 to WAV format first")
                        st.info("• Ensure audio is clear and not corrupted")
                        st.info("• Try a different audio file")
                        return

                    progress_bar.progress(100)
                    status.text("✅ Done!")

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
                                AUDIO_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "wav"))
                                audio_filename = os.path.basename(result['path'])
                                audio_file_path = os.path.join(AUDIO_BASE_DIR, audio_filename)
                                if os.path.exists(audio_file_path):
                                    try:
                                        with open(audio_file_path, 'rb') as f:
                                            st.audio(f.read(), format="audio/wav")
                                    except Exception as e:
                                        st.error(f"Error playing audio: {str(e)}")
                                else:
                                    st.error("Audio file not found: {audio_file_path}")

                    st.subheader("📊 Summary")
                    distances = [r["distance"] for r in results]
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Best Distance", f"{min(distances):.3f}")
                    col2.metric("Average", f"{np.mean(distances):.3f}")
                    col3.metric("Best Similarity", f"{1 - min(distances):.3f}")

            except Exception as e:
                st.error(f"❌ Error processing audio: {str(e)}")
                st.info("**Debug Information:**")
                st.info(f"• Audio source: {'Recorded' if use_recorded else 'Uploaded'}")
                st.info(f"• Temp file exists: {os.path.exists(temp_path) if temp_path else 'No temp file'}")

            finally:
                if temp_path and os.path.exists(temp_path) and not use_recorded:
                    os.remove(temp_path)

def text_learning_app():
    """Main function for the Cree Language Text Learning App.
    
    Provides 3 features:
    1. Translate between Cree and English
    2. Interactive multiple-choice exercises
    3. Dataset explorer for Cree word lookup
    """
    st.header("📝 Cree Language Text Learning")
    
    # Load the model
    model = load_cree_model()
    if model is None:
        st.error("Failed to load the Cree learning model. Please check if the model file exists.")
        return
    
    # Sidebar for text app
    with st.sidebar:
        st.header("Text Learning Mode")
        mode = st.selectbox("Choose Mode", [
            "Translate", "Exercise", "Dataset Explorer", "Add New Entry"])

    # --- Translate Mode ---
    if mode == "Translate":
        st.subheader("🔁 Translation")
        direction = st.radio("Translation Direction", ["Cree → English", "English → Cree"])
        input_word = st.text_input("Enter word:")
        if st.button("Translate"):
            try:
                if direction == "Cree → English":
                    translations = model.find_translations(input_word)
                else:
                    translations = model.find_cree_words(input_word)
                st.write("### Translations:", translations or "No match found.")
            except Exception as e:
                st.error(f"Error during translation: {str(e)}")

    # --- Exercise Mode ---
    elif mode == "Exercise":
        st.subheader("🧪 Take a Cree Translation Test")

        difficulty = st.selectbox("Choose difficulty:", ["mixed", "easy", "hard"])

        # --- Reset on restart or first time ---
        if "text_test_initialized" not in st.session_state or st.session_state.get("text_difficulty_mode") != difficulty:
            try:
                st.session_state.text_test_questions = model.create_learning_exercises(difficulty=difficulty)[:10]
                st.session_state.text_current_q = 0
                st.session_state.text_submitted_answers = [None] * 10
                st.session_state.text_correct_flags = [False] * 10
                st.session_state.text_feedbacks = [""] * 10
                st.session_state.text_finished = False
                st.session_state.text_difficulty_mode = difficulty
                st.session_state.text_test_initialized = True
            except Exception as e:
                st.error(f"Error creating exercises: {str(e)}")
                return

        q_idx = st.session_state.text_current_q
        question = st.session_state.text_test_questions[q_idx]

        # --- If test is not finished ---
        if not st.session_state.text_finished:
            st.markdown(f"### Question {q_idx + 1} of 10")
            st.markdown(f"**Translate this Cree word:** `{question['cree_word']}`")

            selected = st.radio(
                "Choose your answer:",
                options=question["choices"],
                key=f"text_choice_q{q_idx}"
            )

            already_submitted = st.session_state.text_submitted_answers[q_idx] is not None
            if already_submitted:
                st.info(st.session_state.text_feedbacks[q_idx])

            col1, col2 = st.columns(2)

            with col1:
                if st.button("✅ Submit Answer", key="text_submit"):
                    if not already_submitted:
                        st.session_state.text_submitted_answers[q_idx] = selected
                        if selected in question["correct_answers"]:
                            st.session_state.text_correct_flags[q_idx] = True
                            st.session_state.text_feedbacks[q_idx] = "✅ Correct!"
                        else:
                            correct = ", ".join(question["correct_answers"])
                            st.session_state.text_feedbacks[q_idx] = f"❌ Incorrect. Correct answer: **{correct}**"
                        st.rerun()

            with col2:
                if already_submitted and st.button("➡️ Next Question", key="text_next"):
                    if q_idx < 9:
                        st.session_state.text_current_q += 1
                        st.rerun()
                    else:
                        st.warning("This is the last question.")

            st.markdown("---")
            if st.button("🏁 Evaluate Test", key="text_evaluate"):
                st.session_state.text_finished = True
                st.rerun()

        # --- When test is finished ---
        else:
            total_correct = sum(st.session_state.text_correct_flags)
            attempted = sum(ans is not None for ans in st.session_state.text_submitted_answers)

            st.success(f"✅ You answered {total_correct} out of 10 questions correctly.")
            st.write(f"🧮 You attempted {attempted} questions.")
            st.write(f"📊 Your score: **{total_correct} / 10**")
            st.markdown("---")

            for i, q in enumerate(st.session_state.text_test_questions):
                user_answer = st.session_state.text_submitted_answers[i]
                correct = ", ".join(q["correct_answers"])
                st.markdown(f"**Q{i+1}.** `{q['cree_word']}` → Your answer: `{user_answer or 'Not answered'}`")
                if user_answer is None:
                    st.markdown("🟡 Not attempted")
                elif user_answer in q["correct_answers"]:
                    st.markdown("✅ Correct!")
                else:
                    st.markdown(f"❌ Incorrect. Correct answer: **{correct}**")
                st.markdown("—")

            if st.button("🔁 Restart Test", key="text_restart"):
                for key in ["text_test_initialized", "text_current_q", "text_submitted_answers", "text_correct_flags", "text_feedbacks", "text_finished", "text_difficulty_mode"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

    # --- Dataset Explorer ---
    elif mode == "Dataset Explorer":
        st.subheader("📚 Dataset Overview")
        try:
            cree_words = list(model.cree_to_english.keys())
            word_limit = st.slider("How many Cree words to show:", 5, 1000, 10)
            data = [(w, ", ".join(model.cree_to_english[w])) for w in cree_words[:word_limit]]
            df = pd.DataFrame(data, columns=["Cree Word", "English Meanings"])
            st.dataframe(df)

            st.write("---")
            st.write(f"Total Cree words: {len(model.cree_to_english)}")
            st.write(f"Cree words with multiple meanings: {sum(1 for v in model.cree_to_english.values() if len(v) > 1)}")
            st.write(f"Average meanings per Cree word: {np.mean([len(v) for v in model.cree_to_english.values()]):.2f}")
        except Exception as e:
            st.error(f"Error exploring dataset: {str(e)}")

    # --- Add New Entry Mode ---
    elif mode == "Add New Entry":
        st.subheader("➕ Add New Cree–English Entry")

        cree_input = st.text_input("Enter Cree word:")
        english_input = st.text_input("Enter English translation:")

        if st.button("Add Entry"):
            if not cree_input.strip() or not english_input.strip():
                st.warning("⚠️ Please fill in both Cree and English fields.")
            else:
                try:
                    dataset_path = "data/cleaned/cree_english_text_only.csv"
                    model_path = "models/cree_learning_model.pkl"

                    df = pd.read_csv(dataset_path)

                    # Normalize input
                    cree_input_clean = cree_input.strip().lower()
                    english_input_clean = english_input.strip().lower()

                    is_duplicate  = df[
                        (df['Cree'].str.lower() == cree_input_clean) &
                        (df['English'].str.lower() == english_input_clean)
                    ]

                    if not is_duplicate.empty:
                        st.info("This entry already exists in the dataset.")
                    else:
                        new_row = pd.DataFrame([[cree_input.strip(), english_input.strip()]], columns=["Cree", "English"])
                        df = pd.concat([df, new_row], ignore_index=True)
                        df.to_csv(dataset_path, index=False)
                        st.success("✅ Entry added to dataset.")
                        # Retrain model
                        model = CreeLearningModel()
                        model.retrain_from_csv_and_save(csv_path=dataset_path, save_path=model_path)
                        st.success("🧠 Model retrained and saved.")
                        # Push to GitHub
                        push_to_github()
                        st.success("🚀 Changes pushed to GitHub.")

                        # --- 🚨 New addition: reload app ---
                        st.cache_resource.clear()  # Clear model cache
                        st.success("🔄 Reloading app to reflect updates...")
                        import time
                        time.sleep(2)
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ Error during entry addition or model update: {e}")


def push_to_github():
    from github import InputGitAuthor, Github
    from datetime import datetime
    try:
        token = st.secrets["GITHUB_TOKEN"]
        repo_url = st.secrets["GITHUB_REPO_URL"] 
        author_name = st.secrets["GIT_AUTHOR_NAME"]
        author_email = st.secrets["GIT_AUTHOR_EMAIL"]

        # Extract repo name from URL
        repo_name = repo_url.split("/")[-1].replace(".git", "")
        username = repo_url.split("/")[-2]

        g = Github(token)
        repo = g.get_repo(f"{username}/{repo_name}")
        author = InputGitAuthor(author_name, author_email)

        # --- Files to push (csv is text, pkl is binary) ---
        files = {
            "data/cleaned/cree_english_text_only.csv": "text",
            "models/cree_learning_model.pkl": "binary"
        }

        for file_path, file_type in files.items():
            with open(file_path, "rb") as f:
                content_bytes = f.read()

            if file_type == "text":
                encoded_content = content_bytes.decode("utf-8")   # for CSV or text
            else:
                encoded_content = content_bytes        # for PKL (binary)


            try:
                remote_file = repo.get_contents(file_path)
                # Check if file exists → update
                repo.update_file(
                    path=file_path,
                    message=f"🔄 Update {file_path} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    content=encoded_content,
                    sha=remote_file.sha,
                    author=author,
                    branch="main"
                )
                st.success(f"✅ Updated `{file_path}` on GitHub.")
            except Exception as e:
                # If not exists → create
                repo.create_file(
                    path=file_path,
                    message=f"📦 Create {file_path} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    content=encoded_content,
                    author=author,
                    branch="main"
                )
                st.success(f"✅ Created `{file_path}` on GitHub.")
    except Exception as e:
        st.error(f"❌ Error pushing to GitHub: {e}")


def main():
    """Main app with navigation between text and audio learning modes"""
    st.title("🌿 Cree Language Learning App")
    
    # Create navigation bar at the top
    col1, col2, col3 = st.columns([1, 1, 4])
    
    with col1:
        if st.button("📝 Text", use_container_width=True, type="primary" if st.session_state.get('current_tab', 'text') == 'text' else "secondary"):
            st.session_state.current_tab = "text"
    
    with col2:
        if st.button("🎵 Audio", use_container_width=True, type="primary" if st.session_state.get('current_tab', 'text') == 'audio' else "secondary"):
            st.session_state.current_tab = "audio"
    
    # Initialize session state if not exists
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "text"
    
    # Add a horizontal line to separate navigation from content
    st.markdown("---")
    
    # Display the selected app based on the current tab
    if st.session_state.current_tab == "text":
        text_learning_app()
    elif st.session_state.current_tab == "audio":
        audio_learning_app()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            <p>Cree Language Learning System</p>
            <p>Built with Streamlit 🌿</p>
        </div>
        """, unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()