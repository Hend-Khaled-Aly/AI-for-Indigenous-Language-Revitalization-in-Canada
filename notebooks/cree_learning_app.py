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

WEBRTC_AVAILABLE = False


# Set page config
st.set_page_config(
    page_title="Cree Language Learning App",
    page_icon="🌿",
    layout="wide"
)

# Constants for audio app - Fixed paths for cloud deployment
MODELS_DIR = "models/audio"
FEATURES_PATH = os.path.join(MODELS_DIR, "features.npy")
KNN_MODEL_PATH = os.path.join(MODELS_DIR, "knn_model.pkl")
LABELS_PATH = os.path.join(MODELS_DIR, "labels.json")
PATHS_PATH = os.path.join(MODELS_DIR, "paths.json")

# Global variable to store recorded audio
if 'recorded_audio' not in st.session_state:
    st.session_state.recorded_audio = None

# Audio recording class
class AudioRecorder:
    def __init__(self):
        self.frames = []
        self.sample_rate = 16000  # 16kHz for Whisper compatibility
        
    def process_audio_frame(self, frame):
        """Process each audio frame from the recorder"""
        sound = frame.to_ndarray().astype(np.float32)
        # Convert to mono if stereo
        if len(sound.shape) > 1:
            sound = sound.mean(axis=1)
        self.frames.extend(sound)
        return frame
    
    def get_audio_data(self):
        """Get the recorded audio as numpy array"""
        if self.frames:
            return np.array(self.frames, dtype=np.float32)
        return None
    
    def save_as_wav(self, filename):
        """Save recorded audio as WAV file"""
        if self.frames:
            audio_data = np.array(self.frames, dtype=np.float32)
            # Normalize audio
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            sf.write(filename, audio_data, self.sample_rate)
            return True
        return False
    
    def clear(self):
        """Clear recorded frames"""
        self.frames = []

# Helper function to check if files exist
def check_audio_files():
    """Check if all required audio model files exist"""
    required_files = [FEATURES_PATH, KNN_MODEL_PATH, LABELS_PATH, PATHS_PATH]
    missing_files = []
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    return len(missing_files) == 0, missing_files

# Cache functions for audio app
@st.cache_resource
def load_whisper_model():
    """Load and cache the Whisper model with error handling"""
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
    """Load the pre-computed features and metadata with error handling"""
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

# Cache function for text app
@st.cache_resource
def load_cree_model():
    """Load and cache the Cree learning model with error handling"""
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
    """Load and preprocess audio file with soundfile instead of torchaudio"""
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
    """Extract Whisper embeddings from audio file with error handling"""
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
    """Extract Whisper embeddings from audio numpy array"""
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
    """Find similar audio files with error handling"""
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
    """Find similar audio files from numpy array with error handling"""
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
    """Create HTML audio player with error handling"""
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


def audio_learning_app():
    """Audio Learning/Matching App with recording capability"""
    # Sidebar for audio page selection
    with st.sidebar:
        st.header("Audio Options")
        selected_audio_page = st.radio("Choose a view:", ["🎧 Listen to Dataset", "🎙️ Match Audio"])

    # Render based on user selection
    if selected_audio_page == "🎧 Listen to Dataset":
        audio_listening_page()
        return  # Exit to avoid loading models, etc.

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
    """Text Learning App with better error handling"""
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
            "Translate", "Exercise", "Dataset Explorer"])

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

def main():
    """Main app with navigation"""
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