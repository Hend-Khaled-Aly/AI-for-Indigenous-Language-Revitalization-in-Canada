# cree_app.py
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from cree_learning_model import CreeLearningModel

# Set Streamlit page configuration: title and layout style (wide screen)
st.set_page_config(page_title="Cree Language Learning App", layout="wide")

# Display the main title of the app with an emoji for friendliness
st.title("🌿 Cree Language Learning App")

# Cache the model loading function to avoid reloading on every interaction
@st.cache_resource
def load_model():
    # Initialize the CreeLearningModel class instance
    model = CreeLearningModel()
    # Load the pre-trained model from the saved pickle file
    model.load_model("../models/cree_learning_model.pkl")
    return model

# Load the model once and reuse across app interactions
model = load_model()


# Sidebar
mode = st.sidebar.selectbox("Choose Mode", [
    "Translate", "Exercise", "Dataset Explorer"])

# --- Translate Mode ---
if mode == "Translate":
    # Display a header for the translation section with an icon
    st.header("🔁 Translation")
    
    # Allow user to select translation direction: Cree to English or English to Cree
    direction = st.radio("Translation Direction", ["Cree → English", "English → Cree"])
    
    # Text input box for the user to enter the word they want to translate
    input_word = st.text_input("Enter word:")
    
    # A button to trigger the translation action
    if st.button("Translate"):
        # If translating from Cree to English
        if direction == "Cree → English":
            translations = model.find_translations(input_word)
        # If translating from English to Cree
        else:
            translations = model.find_cree_words(input_word)
        
        # Display the translations; if none found, show a friendly message
        st.write("### Translations:", translations or "No match found.")

# --- Exercise Mode ---
elif mode == "Exercise":
    st.header("🧪 Take a Cree Translation Test")
    # User selects difficulty level: mixed, easy, or hard
    difficulty = st.selectbox("Choose difficulty:", ["mixed", "easy", "hard"])

    # --- Initialize or reset test state ---
    # If test is not yet initialized or difficulty changed, reset session state
    if "test_initialized" not in st.session_state or st.session_state.get("difficulty_mode") != difficulty:
        # Generate 10 exercises for the chosen difficulty
        st.session_state.test_questions = model.create_learning_exercises(difficulty=difficulty)[:10]
        st.session_state.current_q = 0  # current question index
        st.session_state.submitted_answers = [None] * 10  # store user answers
        st.session_state.correct_flags = [False] * 10  # track correctness
        st.session_state.feedbacks = [""] * 10  # feedback messages per question
        st.session_state.finished = False  # test finished flag
        st.session_state.difficulty_mode = difficulty  # track current difficulty
        st.session_state.test_initialized = True  # mark test as initialized

    q_idx = st.session_state.current_q
    question = st.session_state.test_questions[q_idx]

    # --- If test is not finished ---
    if not st.session_state.finished:
        st.markdown(f"### Question {q_idx + 1} of 10")
        st.markdown(f"**Translate this Cree word:** `{question['cree_word']}`")
        # Show multiple choice options as radio buttons
        selected = st.radio(
            "Choose your answer:",
            options=question["choices"],
            key=f"choice_q{q_idx}"
        )
        # Show feedback if already answered
        already_submitted = st.session_state.submitted_answers[q_idx] is not None
        if already_submitted:
            st.info(st.session_state.feedbacks[q_idx])

        col1, col2 = st.columns(2)

        with col1:
            if st.button("✅ Submit Answer"):
                if not already_submitted:
                    st.session_state.submitted_answers[q_idx] = selected
                    # Check correctness
                    if selected in question["correct_answers"]:
                        st.session_state.correct_flags[q_idx] = True
                        st.session_state.feedbacks[q_idx] = "✅ Correct!"
                    else:
                        correct = ", ".join(question["correct_answers"])
                        st.session_state.feedbacks[q_idx] = f"❌ Incorrect. Correct answer: **{correct}**"
                    st.rerun()

        with col2:
            # Next question button (only after submission)
            if already_submitted and st.button("➡️ Next Question"):
                if q_idx < 9:
                    st.session_state.current_q += 1
                    st.rerun()
                else:
                    st.warning("This is the last question.")

        st.markdown("---")
        # finish and evaluate the test
        if st.button("🏁 Evaluate Test"):
            st.session_state.finished = True
            st.rerun()

    # --- When test is finished ---
    else:
        total_correct = sum(st.session_state.correct_flags)
        attempted = sum(ans is not None for ans in st.session_state.submitted_answers)

        st.success(f"✅ You answered {total_correct} out of 10 questions correctly.")
        st.write(f"🧮 You attempted {attempted} questions.")
        st.write(f"📊 Your score: **{total_correct} / 10**")
        st.markdown("---")
        # Show detailed feedback for all questions
        for i, q in enumerate(st.session_state.test_questions):
            user_answer = st.session_state.submitted_answers[i]
            correct = ", ".join(q["correct_answers"])
            st.markdown(f"**Q{i+1}.** `{q['cree_word']}` → Your answer: `{user_answer or 'Not answered'}`")
            if user_answer is None:
                st.markdown("🟡 Not attempted")
            elif user_answer in q["correct_answers"]:
                st.markdown("✅ Correct!")
            else:
                st.markdown(f"❌ Incorrect. Correct answer: **{correct}**")
            st.markdown("—")
        # restart the test and reset all relevant session state
        if st.button("🔁 Restart Test"):
            for key in ["test_initialized", "current_q", "submitted_answers", "correct_flags", "feedbacks", "finished", "difficulty_mode"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

# --- Dataset Explorer ---
elif mode == "Dataset Explorer":
    # Display header for dataset overview
    st.header("📚 Dataset Overview")
    # Get all Cree words from the model's dictionary
    cree_words = list(model.cree_to_english.keys())
    # Allow user to select how many Cree words to display using a slider (min=5, max=1000, default=10)
    word_limit = st.slider("How many Cree words to show:", 5, 1000, 10)
    # Prepare data for display: a list of tuples with (Cree word, English meanings joined by commas)
    data = [(w, ", ".join(model.cree_to_english[w])) for w in cree_words[:word_limit]]
     # Create a Pandas DataFrame from the data with appropriate column names
    df = pd.DataFrame(data, columns=["Cree Word", "English Meanings"])
    # Show the dataframe interactively in the Streamlit app
    st.dataframe(df)

    st.write("---")
    # Display summary statistics about the dataset
    st.write(f"Total Cree words: {len(model.cree_to_english)}")
    # Count and display how many Cree words have multiple English meanings
    st.write(f"Cree words with multiple meanings: {sum(1 for v in model.cree_to_english.values() if len(v) > 1)}")
    # Calculate and display the average number of English meanings per Cree word (formatted to 2 decimals)
    st.write(f"Average meanings per Cree word: {np.mean([len(v) for v in model.cree_to_english.values()]):.2f}")