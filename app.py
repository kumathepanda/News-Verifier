import streamlit as st
import joblib
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import time

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r"[^a-zA-Z]", ' ', text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    return " ".join(tokens)

# Prediction function with probabilities
def predict_news(text):
    clean_text = preprocess_text(text)
    vector = vectorizer.transform([clean_text])
    prediction = model.predict(vector)[0]
    probabilities = model.predict_proba(vector)[0]
    
    # Assuming 0 = Fake, 1 = Real
    fake_prob = probabilities[0] * 100
    real_prob = probabilities[1] * 100
    
    return prediction, fake_prob, real_prob

# Page configuration
st.set_page_config(
    page_title="News Verifier",
    page_icon="📰",
    layout="centered"
)

# Simple CSS with peach/newspaper theme
st.markdown("""
    <style>
        /* Import classic fonts */
        @import url('https://fonts.googleapis.com/css2?family=Crimson+Text:wght@400;600&family=Source+Serif+Pro:wght@400;600&display=swap');
        
        /* Main styling */
        .stApp {
            background-color: #fdf6e3;
            color: #2c1810;
        }
        
        /* Hide Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Header */
        .main-header {
            text-align: center;
            padding: 2rem 0;
            border-bottom: 3px solid #d4a574;
            margin-bottom: 2rem;
        }
        
        .main-title {
            font-family: 'Crimson Text', serif;
            font-size: 2.5rem;
            font-weight: 600;
            color: #8b4513;
            margin: 0;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        }
        
        .subtitle {
            font-family: 'Source Serif Pro', serif;
            font-size: 1rem;
            color: #a0522d;
            margin-top: 0.5rem;
            font-style: italic;
        }
        
        /* Input area */
        .input-container {
            background-color: #faf0e6;
            padding: 1.5rem;
            border-radius: 8px;
            border: 2px solid #deb887;
            margin: 1rem 0;
        }
        
        .input-label {
            font-family: 'Crimson Text', serif;
            font-size: 1.2rem;
            color: #8b4513;
            margin-bottom: 1rem;
            font-weight: 600;
        }
        
        /* Button styling */
        .stButton > button {
            background-color: #cd853f;
            color: white;
            border: none;
            padding: 0.7rem 2rem;
            border-radius: 5px;
            font-family: 'Source Serif Pro', serif;
            font-size: 1rem;
            font-weight: 600;
            width: 100%;
            margin-top: 1rem;
            transition: background-color 0.3s;
        }
        
        .stButton > button:hover {
            background-color: #a0522d;
        }
        
        /* Results */
        .result-box {
            padding: 1.5rem;
            border-radius: 8px;
            margin: 1.5rem 0;
            text-align: center;
            font-family: 'Source Serif Pro', serif;
        }
        
        .result-real {
            background-color: #f0f8f0;
            border: 2px solid #228b22;
            color: #006400;
        }
        
        .result-fake {
            background-color: #fdf0f0;
            border: 2px solid #cd5c5c;
            color: #8b0000;
        }
        
        .result-title {
            font-size: 1.4rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        
        .probability-section {
            margin-top: 1rem;
            padding-top: 1rem;
            border-top: 1px solid rgba(0,0,0,0.1);
        }
        
        .prob-bar {
            background-color: #e0e0e0;
            border-radius: 10px;
            height: 20px;
            margin: 0.5rem 0;
            overflow: hidden;
        }
        
        .prob-fill-real {
            background-color: #228b22;
            height: 100%;
            border-radius: 10px;
            transition: width 0.5s ease;
        }
        
        .prob-fill-fake {
            background-color: #cd5c5c;
            height: 100%;
            border-radius: 10px;
            transition: width 0.5s ease;
        }
        
        .prob-text {
            font-size: 0.9rem;
            margin: 0.2rem 0;
        }
        
        /* Text area */
        .stTextArea textarea {
            font-family: 'Source Serif Pro', serif;
            border: 1px solid #deb887;
            border-radius: 5px;
            background-color: #fffef7;
            color: #2c1810 !important;
            font-size: 14px;
            cursor: text;
        }
        
        .stTextArea textarea::placeholder {
            color: #8b7355 !important;
            opacity: 0.7;
        }
        
        .stTextArea textarea:focus {
            border-color: #cd853f;
            outline: none;
            box-shadow: 0 0 5px rgba(205, 133, 63, 0.3);
            cursor: text;
        }
        
        .stTextArea textarea:focus::placeholder {
            opacity: 0.3;
        }
        
        /* Warning */
        .stWarning {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 5px;
        }
        
        /* Footer */
        .footer-note {
            margin-top: 2rem;
            padding: 1rem;
            background-color: #f5f5dc;
            border-radius: 5px;
            border-left: 4px solid #cd853f;
            font-family: 'Source Serif Pro', serif;
            font-size: 0.9rem;
            color: #654321;
            font-style: italic;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="main-header">
        <h1 class="main-title">📰 News Verifier</h1>
        <p class="subtitle">Authenticate news content with AI analysis</p>
    </div>
""", unsafe_allow_html=True)

# Input section
st.markdown("""
    <div class="input-container">
        <div class="input-label">📝 Enter news content to verify:</div>
    </div>
""", unsafe_allow_html=True)

input_text = st.text_area(
    "Text to analyze:",
    height=150,
    placeholder="Paste your news article, headline, or any text content here...",
    label_visibility="collapsed",
    key="news_input"
)

# Analyze button
if st.button("🔍 Analyze Content"):
    if input_text.strip() == "":
        st.warning("⚠️ Please enter some content to analyze.")
    else:
        # Show loading
        with st.spinner('📊 Analyzing content...'):
            time.sleep(1)
            prediction, fake_prob, real_prob = predict_news(input_text)
        
        # Display results
        if prediction == 0:  # Fake
            st.markdown(f"""
                <div class="result-box result-fake">
                    <div class="result-title">🚨 LIKELY FAKE NEWS</div>
                    <p>This content shows characteristics of misleading information.</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Probability display using Streamlit components
            st.markdown("**Confidence Levels:**")
            st.write(f"🚨 Fake: **{fake_prob:.1f}%**")
            st.progress(fake_prob/100)
            st.write(f"✅ Authentic: **{real_prob:.1f}%**")
            st.progress(real_prob/100)
            
        else:  # Real
            st.markdown(f"""
                <div class="result-box result-real">
                    <div class="result-title">✅ LIKELY AUTHENTIC</div>
                    <p>This content appears to follow patterns of legitimate news.</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Probability display using Streamlit components
            st.markdown("**Confidence Levels:**")
            st.write(f"✅ Authentic: **{real_prob:.1f}%**")
            st.progress(real_prob/100)
            st.write(f"🚨 Fake: **{fake_prob:.1f}%**")
            st.progress(fake_prob/100)

# Footer note
st.markdown("""
    <div class="footer-note">
        <strong>📌 Important:</strong> This tool provides AI-based guidance. Always verify important information through multiple trusted sources before making decisions.
    </div>
""", unsafe_allow_html=True)