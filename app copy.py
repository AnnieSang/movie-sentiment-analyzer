# pip install streamlit
# How to run
# streamlit run app.py

import streamlit as st
from transformers import pipeline

# 1. Page Config
st.set_page_config(page_title="Sentiment AI", layout="wide")

# 2. Custom CSS
st.markdown("""
    <style>
    .block-container { padding-top: 2rem; }
    /* Centered Title Styling */
    .stTitle {
        text-align: center;
        width: 100%;
    }
    /* Center the text inside the result box */
    div[data-testid="stNotification"] {
        text-align: center;
    }
    /* Button styling to fill their columns */
    .stButton button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Load Model using Hugging Face Pipeline
@st.cache_resource
def load_model():
    # Ensure this path points to your saved model/tokenizer folder
    return pipeline("sentiment-analysis", model="./my_imdb_model")

classifier = load_model()

# 4. Session State Management
if "user_text" not in st.session_state:
    st.session_state.user_text = ""
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

def clear_text():
    st.session_state.user_text = ""
    st.session_state.analysis_result = None

# --- UI LAYOUT ---

# Step 1: Main Content Column (Ratio 1-10-1 keeps things centered and wide)
_, main_col, _ = st.columns([1, 10, 1])

with main_col:
    # Title aligned with the rest of the items
    st.title("🎬 Movie Review Sentiment Analyzer")
    
    # Input area
    user_input = st.text_area(
        "Enter your review:", 
        value=st.session_state.user_text, 
        height=150,
        key="text_input_field"
    )

    # Buttons Row (Grouped together under the text area)
    btn_col1, btn_col2, btn_spacer = st.columns([1.5, 1.5, 7])
    
    with btn_col1:
        if st.button("Analyze", type="primary"):
            if user_input.strip():
                # [0] fixes the 'list indices must be integers' error
                result = classifier(user_input)[0] 
                st.session_state.analysis_result = result
                st.session_state.user_text = user_input
            else:
                st.warning("Please enter some text first.")
                
    with btn_col2:
        if st.button("Clear Text", on_click=clear_text):
            # Clears the state and refreshes the UI
            st.rerun()

# Step 2: Result Box (Ratio 3-4-3 keeps the green/red box less wide)
st.write("") # Vertical spacer
_, res_col, _ = st.columns([3, 4, 3])

with res_col:
    if st.session_state.analysis_result:
        res = st.session_state.analysis_result
        label = res['label']
        score = res['score']
        
        # Logic to handle model labels
        sentiment = "POSITIVE" if label == "LABEL_1" else "NEGATIVE"
        
        if sentiment == "POSITIVE":
            st.success(f"✅ **{sentiment}** (Confidence: {score:.2%})")
        else:
            st.error(f"❌ **{sentiment}** (Confidence: {score:.2%})")
