import streamlit as st
import pickle
import os

# --- LOAD THE TRAINED MODELS ---
@st.cache_resource
def load_models():
    """Loads the trained vectorizer and model from disk."""
    if not os.path.exists("vectorizer.pkl") or not os.path.exists("spam_model.pkl"):
        st.error("Model files not found! Please make sure 'vectorizer.pkl' and 'spam_model.pkl' are in the same folder as 'app.py'.")
        st.stop() # Stop the app if files are missing

    try:
        with open("vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        with open("spam_model.pkl", "rb") as f:
            model = pickle.load(f)
        return vectorizer, model
    except Exception as e:
        st.error(f"An error occurred while loading the model files: {e}")
        st.stop()

vectorizer, model = load_models()

# --- APP LAYOUT AND STYLING ---
st.set_page_config(page_title="Email Spam Classifier", page_icon="📧", layout="wide")

# Custom CSS for a more polished look
st.markdown("""
<style>
    .main { background-color: #f5f5f5; }
    .stButton>button { background-color: #4CAF50; color: white; border-radius: 12px; border: 1px solid #4CAF50; padding: 10px 24px; text-align: center; font-size: 16px; margin: 4px 2px; cursor: pointer; transition-duration: 0.4s; }
    .stButton>button:hover { background-color: white; color: black; }
    .stTextArea textarea { border-radius: 12px; border: 2px solid #ccc; padding: 10px; }
    .result-ham { color: green; font-weight: bold; font-size: 24px; border: 2px solid green; padding: 15px; border-radius: 12px; text-align: center; }
    .result-spam { color: red; font-weight: bold; font-size: 24px; border: 2px solid red; padding: 15px; border-radius: 12px; text-align: center; }
</style>
""", unsafe_allow_html=True)

# --- HEADER SECTION ---
st.title("📧 Email Spam Classifier")
st.markdown("Enter an email text below to check if it's **Spam** or **Ham** (not spam).")
st.markdown("---")

# --- MAIN INTERFACE ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Enter Email Content")
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""
    user_input = st.text_area("Email Text:", height=250, key="user_input_area", value=st.session_state.user_input, placeholder="Paste your email content here...")
    button_col1, button_col2, _ = st.columns([1, 1, 3])
    classify_button = button_col1.button("Classify", use_container_width=True)
    clear_button = button_col2.button("Clear", use_container_width=True)
    if clear_button:
        st.session_state.user_input = ""
        st.rerun()

with col2:
    st.subheader("Classification Result")
    with st.expander("See an example"):
        st.write("""**Subject: Congratulations! You've Won!**\n\nClick the link to claim your prize...""")
    if classify_button:
        if user_input.strip() == "":
            st.warning("⚠️ Please enter some text to classify.")
        else:
            user_vec = vectorizer.transform([user_input])
            prediction = model.predict(user_vec)[0]
            if prediction == 0:
                st.markdown('<p class="result-ham">✅ This looks like Ham (Not Spam)!</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="result-spam">🚨 This looks like Spam!</p>', unsafe_allow_html=True)
    else:
        st.info("The classification result will appear here.")
st.markdown("---")
st.write("Powered by a Multinomial Naive Bayes model.")

