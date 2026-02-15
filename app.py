# pip install streamlit
# How to run
# streamlit run app.py

import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Sentiment AI", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .block-container { padding-top: 2rem; }
    .stTitle { text-align: center; width: 100%; }
    div[data-testid="stNotification"] { text-align: center; }
    .stButton button { width: 100%; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="./my_imdb_model")

classifier = load_model()

if "user_text" not in st.session_state:
    st.session_state.user_text = ""
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

# Updated Clear Function
def clear_text():
    st.session_state.user_text = ""
    st.session_state.analysis_result = None
    if "text_input_field" in st.session_state:
        st.session_state.text_input_field = ""

# --- UI LAYOUT ---
_, main_col, _ = st.columns([1, 10, 1])

with main_col:
    st.title("🎬 Movie Review Sentiment Analyzer")
    
    user_input = st.text_area(
        "Enter your review:", 
        value=st.session_state.user_text, 
        height=150,
        key="text_input_field"
    )

    btn_col1, btn_col2, btn_spacer = st.columns([1.5, 1.5, 7])
    
    with btn_col1:
        if st.button("Analyze", type="primary"):
            if user_input.strip():
                result = classifier(user_input)[0] 
                st.session_state.analysis_result = result
                st.session_state.user_text = user_input
            else:
                st.warning("Please enter some text first.")
                
    with btn_col2:
        # Use on_click to trigger the reset before the page reruns
        st.button("Clear Text", on_click=clear_text)

# Result Box
st.write("") 
_, res_col, _ = st.columns([3, 4, 3])

with res_col:
    if st.session_state.analysis_result:
        res = st.session_state.analysis_result
        label = res['label']
        score = res['score']
        sentiment = "POSITIVE" if label == "LABEL_1" else "NEGATIVE"
        
        if sentiment == "POSITIVE":
            st.success(f"✅ **{sentiment}** (Confidence: {score:.2%})")
        else:
            st.error(f"❌ **{sentiment}** (Confidence: {score:.2%})")
