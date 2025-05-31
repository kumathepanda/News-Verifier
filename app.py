import streamlit as st
import joblib
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import time
import pandas as pd
import os
from datetime import datetime
import json
import gspread
from google.oauth2.service_account import Credentials

# Page configuration - MUST BE FIRST
st.set_page_config(
    page_title="News Verifier",
    page_icon="📰",
    layout="centered"
)

# GOOGLE SHEETS CONFIGURATION
GOOGLE_SHEETS_URL = "https://docs.google.com/spreadsheets/d/16B6LHV0CakAfH2JOgxFv8F0Dv86_sMfCII5wGWvPYnk/edit?usp=sharing"  
SHEET_NAME = "feedback_data"  

# Initialize session state variables
if 'feedback_submitted' not in st.session_state:
    st.session_state.feedback_submitted = False
if 'show_success' not in st.session_state:
    st.session_state.show_success = False
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'current_prediction' not in st.session_state:
    st.session_state.current_prediction = None
if 'current_text' not in st.session_state:
    st.session_state.current_text = ""

# Download NLTK data with comprehensive error handling
@st.cache_resource
def download_nltk_data():
    """Download required NLTK data with caching"""
    try:
        # Try to find existing data first
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
        return True
    except LookupError:
        try:
            # Download required data
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            # Try alternative punkt tokenizer for newer NLTK versions
            nltk.download('punkt_tab', quiet=True)
            return True
        except Exception as e:
            st.error(f"Failed to download NLTK data: {e}")
            return False

# Initialize NLTK data
download_nltk_data()

# Load the model and vectorizer
@st.cache_resource
def load_models():
    """Load ML models with caching"""
    try:
        model = joblib.load("fake_news_model.pkl")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        return model, vectorizer
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return None, None

model, vectorizer = load_models()

# Preprocessing setup
@st.cache_resource
def setup_preprocessing():
    """Setup preprocessing tools with caching"""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    return lemmatizer, stop_words

lemmatizer, stop_words = setup_preprocessing()

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
    if model is None or vectorizer is None:
        st.error("Models not loaded properly. Please refresh the page.")
        return None, None, None
        
    clean_text = preprocess_text(text)
    vector = vectorizer.transform([clean_text])
    prediction = model.predict(vector)[0]
    probabilities = model.predict_proba(vector)[0]
    
    # Assuming 0 = Fake, 1 = Real
    fake_prob = probabilities[0] * 100
    real_prob = probabilities[1] * 100
    
    return prediction, fake_prob, real_prob

@st.cache_resource
def setup_google_sheets():
    """Setup Google Sheets client with caching - Fixed for Streamlit secrets"""
    try:
        # Define the scope
        scope = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive'
        ]
        
        # Try to load credentials from Streamlit secrets first
        try:
            # Access the gcp_service_account from Streamlit secrets
            credentials_dict = st.secrets["gcp_service_account"]
            
            # Convert the AttrDict to a regular dictionary
            credentials_info = dict(credentials_dict)
            
            # Fix the private_key formatting (replace \\n with actual newlines)
            if "private_key" in credentials_info:
                credentials_info["private_key"] = credentials_info["private_key"].replace("\\n", "\n")
            
            # Create credentials from the dictionary
            creds = Credentials.from_service_account_info(credentials_info, scopes=scope)
            
        except Exception as secrets_error:
            # Fallback: Try to load from local file
            if os.path.exists("service_account.json"):
                creds = Credentials.from_service_account_file("service_account.json", scopes=scope)
            else:
                return None, f"Credentials not found. Secrets error: {str(secrets_error)}"
        
        # Authorize and create client
        client = gspread.authorize(creds)
        return client, "success"
        
    except Exception as e:
        return None, f"Failed to setup Google Sheets: {str(e)}"

def extract_sheet_id_from_url(url):
    """Extract Google Sheets ID from URL"""
    try:
        if '/spreadsheets/d/' in url:
            return url.split('/spreadsheets/d/')[1].split('/')[0]
        else:
            return url  # Assume it's already an ID
    except Exception:
        return None

@st.cache_data(ttl=60)  # Cache for 1 minute to avoid frequent API calls
def load_google_sheets_data():
    """Load data from Google Sheets with caching"""
    if not GOOGLE_SHEETS_URL or GOOGLE_SHEETS_URL == "YOUR_GOOGLE_SHEETS_URL_HERE":
        return pd.DataFrame(), "Google Sheets URL not configured"
    
    try:
        client, setup_status = setup_google_sheets()
        if setup_status != "success":
            return pd.DataFrame(), setup_status
        
        # Extract sheet ID from URL
        sheet_id = extract_sheet_id_from_url(GOOGLE_SHEETS_URL)
        if not sheet_id:
            return pd.DataFrame(), "Invalid Google Sheets URL"
        
        # Open the spreadsheet
        spreadsheet = client.open_by_key(sheet_id)
        
        # Try to open the specific worksheet
        try:
            worksheet = spreadsheet.worksheet(SHEET_NAME)
        except gspread.WorksheetNotFound:
            # Create the worksheet if it doesn't exist
            worksheet = spreadsheet.add_worksheet(title=SHEET_NAME, rows=1000, cols=10)
            # Add headers
            headers = ['clean_text', 'label', 'timestamp', 'session_id']
            worksheet.insert_row(headers, 1)
        
        # Get all records
        records = worksheet.get_all_records()
        df = pd.DataFrame(records)
        
        return df, "success"
        
    except gspread.exceptions.APIError as e:
        return pd.DataFrame(), f"Google Sheets API error: {str(e)}"
    except Exception as e:
        return pd.DataFrame(), f"Error loading Google Sheets data: {str(e)}"

def save_feedback_to_google_sheets(preprocessed_text, corrected_label):
    """Save feedback data to Google Sheets and local session"""
    try:
        # Create feedback record
        feedback_record = {
            'clean_text': preprocessed_text,
            'label': int(corrected_label),
            'timestamp': datetime.now().isoformat(),
            'session_id': st.session_state.get('session_id', 'unknown')
        }
        
        # Save to session state (immediate backup)
        if 'feedback_data' not in st.session_state:
            st.session_state.feedback_data = []
        st.session_state.feedback_data.append(feedback_record)
        
        # Try to save to Google Sheets
        try:
            client, setup_status = setup_google_sheets()
            if setup_status != "success":
                # Fallback to local save
                df = pd.DataFrame(st.session_state.feedback_data)
                df.to_csv("feedback_data_backup.csv", index=False)
                return True, f"Google Sheets not available ({setup_status}). Saved locally. Records: {len(df)}"
            
            # Extract sheet ID from URL
            sheet_id = extract_sheet_id_from_url(GOOGLE_SHEETS_URL)
            if not sheet_id:
                return False, "Invalid Google Sheets URL"
            
            # Open the spreadsheet and worksheet
            spreadsheet = client.open_by_key(sheet_id)
            
            try:
                worksheet = spreadsheet.worksheet(SHEET_NAME)
            except gspread.WorksheetNotFound:
                # Create the worksheet if it doesn't exist
                worksheet = spreadsheet.add_worksheet(title=SHEET_NAME, rows=1000, cols=10)
                # Add headers
                headers = ['clean_text', 'label', 'timestamp', 'session_id']
                worksheet.insert_row(headers, 1)
            
            # Append the new record
            row_data = [
                feedback_record['clean_text'],
                feedback_record['label'],
                feedback_record['timestamp'],
                feedback_record['session_id']
            ]
            worksheet.append_row(row_data)
            
            # Get total count
            total_records = len(worksheet.get_all_records())
            
            return True, f"Feedback saved to Google Sheets! Total records: {total_records}"
                
        except Exception as sheets_error:
            # Fallback to local CSV
            try:
                df = pd.DataFrame(st.session_state.feedback_data)
                df.to_csv("feedback_data_backup.csv", index=False)
                return True, f"Google Sheets error, saved locally: {sheets_error}. Records: {len(df)}"
            except Exception as csv_error:
                return False, f"Failed to save feedback: {csv_error}"
        
    except Exception as e:
        return False, f"Failed to save feedback: {e}"

def load_feedback_stats():
    """Load and display feedback statistics from Google Sheets and session"""
    stats = {
        'sheets_feedback': 0,
        'session_feedback': 0,
        'total_feedback': 0,
        'sheets_status': 'Not configured'
    }
    
    # Get session feedback count
    stats['session_feedback'] = len(st.session_state.get('feedback_data', []))
    
    # Try to get Google Sheets feedback count
    if GOOGLE_SHEETS_URL and GOOGLE_SHEETS_URL != "YOUR_GOOGLE_SHEETS_URL_HERE":
        try:
            df, load_status = load_google_sheets_data()
            if load_status == "success" and not df.empty:
                stats['sheets_feedback'] = len(df)
                stats['sheets_status'] = 'Connected'
            else:
                stats['sheets_status'] = f'Error: {load_status}'
        except Exception as e:
            stats['sheets_status'] = f'Error: {str(e)}'
    
    # Calculate total (avoid double counting)
    stats['total_feedback'] = max(stats['sheets_feedback'], stats['session_feedback'])
    
    return stats

# Generate unique session ID
if 'session_id' not in st.session_state:
    st.session_state.session_id = f"session_{int(time.time())}"

# CSS with improved input label styling and coffee-colored radio buttons
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Crimson+Text:wght@400;600&family=Source+Serif+Pro:wght@400;600&display=swap');
        
        .stApp {
            background-color: #fdf6e3;
            color: #2c1810;
        }
        
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
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
        
        .input-container {
            background-color: #faf0e6;
            padding: 1.5rem;
            border-radius: 10px;
            border: 2px solid #deb887;
            margin: 1.5rem 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .input-label {
            font-family: 'Crimson Text', serif;
            font-size: 1.3rem;
            color: #8b4513;
            margin-bottom: 1rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.8rem 0;
            border-bottom: 2px solid #e6d3b7;
            background: linear-gradient(135deg, #fff8e7 0%, #f5e6d3 100%);
            border-radius: 8px;
            padding-left: 1rem;
            box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
        }
        
        .input-label::before {
            content: "📝";
            font-size: 1.4rem;
            margin-right: 0.3rem;
        }
        
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
        
        .stTextArea textarea {
            font-family: 'Source Serif Pro', serif;
            border: 2px solid #deb887;
            border-radius: 8px;
            background-color: #fffef7;
            color: #2c1810 !important;
            font-size: 14px;
            cursor: text;
            padding: 0.8rem;
            transition: border-color 0.3s, box-shadow 0.3s;
        }
        
        .stTextArea textarea::placeholder {
            color: #8b7355 !important;
            opacity: 0.7;
        }
        
        .stTextArea textarea:focus {
            border-color: #cd853f;
            outline: none;
            box-shadow: 0 0 8px rgba(205, 133, 63, 0.4);
            cursor: text;
        }
        
        .stTextArea textarea:focus::placeholder {
            opacity: 0.3;
        }
        
        .stWarning {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 5px;
        }
        
        /* Coffee-colored radio buttons styling */
        .stRadio > div {
            background-color: #f9f2e7;
            padding: 1rem;
            border-radius: 8px;
            border: 2px solid #deb887;
            margin: 0.5rem 0;
        }
        
        /* Radio button title styling */
        .stRadio > div > label > div[data-testid="stMarkdownContainer"] > p {
            color: #654321 !important;
            font-family: 'Crimson Text', serif !important;
            font-weight: 600 !important;
            font-size: 1.1rem !important;
        }
        
        /* Radio button options text styling - target multiple selectors */
        .stRadio div[role="radiogroup"] label,
        .stRadio div[role="radiogroup"] label span,
        .stRadio div[role="radiogroup"] label div,
        .stRadio div[role="radiogroup"] label p {
            color: #654321 !important;
            font-family: 'Source Serif Pro', serif !important;
            font-weight: 500 !important;
            font-size: 1rem !important;
        }
        
        /* Radio button container styling */
        .stRadio div[role="radiogroup"] label {
            background-color: #fff8f0 !important;
            padding: 0.8rem 1rem !important;
            border-radius: 8px !important;
            margin: 0.3rem 0 !important;
            border: 2px solid #e6d3b7 !important;
            transition: all 0.3s ease !important;
            display: flex !important;
            align-items: center !important;
        }
        
        .stRadio div[role="radiogroup"] label:hover {
            background-color: #f5e6d3 !important;
            border-color: #cd853f !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 2px 8px rgba(205, 133, 63, 0.2) !important;
        }
        
        /* Ensure radio button text is coffee colored */
        .stRadio [data-testid="stMarkdownContainer"] p {
            color: #654321 !important;
        }
        
        /* Feedback container styling */
        .feedback-container {
            background-color: #f9f2e7;
            padding: 1.5rem;
            border-radius: 10px;
            border: 2px solid #deb887;
            margin: 1.5rem 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .feedback-title {
            font-family: 'Crimson Text', serif;
            font-size: 1.3rem;
            color: #8b4513;
            font-weight: 600;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .feedback-title::before {
            content: "💭";
            font-size: 1.4rem;
        }
        
        .stats-container {
            background-color: #e8f5e8;
            padding: 1rem;
            border-radius: 8px;
            border: 2px solid #90ee90;
            margin: 1rem 0;
            font-family: 'Source Serif Pro', serif;
            color: #2d5a2d;
        }
        
        .sheets-status-connected {
            background-color: #d4edda;
            color: #155724;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            font-weight: 600;
            display: inline-block;
            margin: 0.5rem 0;
        }
        
        .sheets-status-error {
            background-color: #f8d7da;
            color: #721c24;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            font-weight: 600;
            display: inline-block;
            margin: 0.5rem 0;
        }
        
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
        
        /* Toast notification styling */
        .toast-success {
            background-color: #d4edda;
            color: #155724;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #c3e6cb;
            margin: 1rem 0;
            font-family: 'Source Serif Pro', serif;
            font-weight: 500;
            text-align: center;
            animation: slideIn 0.5s ease-out;
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(-20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
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

# Display feedback statistics if available
stats = load_feedback_stats()
if stats:
    sheets_status_class = "sheets-status-connected" if stats['sheets_status'] == 'Connected' else "sheets-status-error"
    
    st.markdown(f"""
        <div class="stats-container">
            <strong>📊 Community Feedback Stats:</strong><br>
            • Total Feedback Received: {stats['total_feedback']}<br>
            • Google Sheets Records: {stats['sheets_feedback']}<br>
            • Your Session Feedback: {stats['session_feedback']}<br>
            • Google Sheets Status: <span class="{sheets_status_class}">{stats['sheets_status']}</span><br>
            • Your feedback helps improve our model for everyone!
        </div>
    """, unsafe_allow_html=True)

# Configuration warning if Google Sheets not set up
if GOOGLE_SHEETS_URL == "YOUR_GOOGLE_SHEETS_URL_HERE":
    st.warning("⚠️ Google Sheets integration not configured. Please set your Google Sheets URL and credentials.")

# Input section
st.markdown("""
    <div class="input-container">
        <div class="input-label">Enter news content to verify</div>
    </div>
""", unsafe_allow_html=True)

input_text = st.text_area(
    "Text to analyze:",
    height=150,
    placeholder="Paste your news article, headline, or any text content here...",
    label_visibility="collapsed",
    key="news_input"
)

if st.button("🔍 Analyze Content"):
    if input_text.strip() == "":
        st.warning("⚠️ Please enter some content to analyze.")
    elif model is None or vectorizer is None:
        st.error("⚠️ Models are not loaded properly. Please refresh the page and try again.")
    else:
        with st.spinner('📊 Analyzing content...'):
            time.sleep(1)
            result = predict_news(input_text)
            
        if result[0] is not None:
            # Store results in session state
            st.session_state.analysis_done = True
            st.session_state.current_prediction = result
            st.session_state.current_text = input_text
            st.session_state.feedback_submitted = False
            st.session_state.show_success = False

# Display results if analysis was done
if st.session_state.analysis_done and st.session_state.current_prediction:
    prediction, fake_prob, real_prob = st.session_state.current_prediction
    
    if prediction == 0:  # Fake
        st.markdown(f"""
            <div class="result-box result-fake">
                <div class="result-title">🚨 LIKELY FAKE NEWS</div>
                <p>This content shows characteristics of misleading information.</p>
            </div>
        """, unsafe_allow_html=True)
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
        st.markdown("**Confidence Levels:**")
        st.write(f"✅ Authentic: **{real_prob:.1f}%**")
        st.progress(real_prob/100)
        st.write(f"🚨 Fake: **{fake_prob:.1f}%**")
        st.progress(fake_prob/100)

    # Enhanced Human Feedback Section
    st.markdown("""
        <div class="feedback-container">
            <div class="feedback-title">Help us improve our model</div>
        </div>
    """, unsafe_allow_html=True)

    # Show success message if feedback was just submitted
    if st.session_state.show_success:
        st.markdown("""
            <div class="toast-success">
                🎉 Thank you for your valuable feedback! Your input has been recorded and will be used to retrain our model in the next update. Together, we're making news verification more reliable!
            </div>
        """, unsafe_allow_html=True)
        st.success("✅ Feedback successfully recorded!")

    # Only show feedback form if not recently submitted
    if not st.session_state.feedback_submitted:
        feedback = st.radio(
            "Was our prediction incorrect? Your feedback helps us improve:",
            options=["No Feedback", "It was Real News", "It was Fake News"],
            index=0,
            horizontal=True,
            key="feedback_radio"
        )

        # Submit button for feedback - only show when feedback is selected
        if feedback != "No Feedback":
            st.markdown("---")  # Add a separator
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                submit_clicked = st.button("📤 Submit Feedback", key="submit_feedback")
            
            # Only process feedback when submit button is clicked
            if submit_clicked:
                corrected_label = 1 if feedback == "It was Real News" else 0
                clean_text = preprocess_text(st.session_state.current_text)
                
                # Save feedback to Google Sheets and session
                success, message = save_feedback_to_google_sheets(
                    preprocessed_text=clean_text,
                    corrected_label=corrected_label
                )
                
                if success:
                    # Set session state to show success message
                    st.session_state.feedback_submitted = True
                    st.session_state.show_success = True
                    # Clear cache to refresh stats
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error(f"⚠️ {message}")
            
            # Show instruction text when feedback is selected but not yet submitted
            else:
                st.info("👆 Please click 'Submit Feedback' to confirm your selection and help improve our model.")
    else:
        # Show a message that feedback was received and allow new feedback
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔄 Provide New Feedback", key="new_feedback"):
                st.session_state.feedback_submitted = False
                st.session_state.show_success = False
                st.rerun()

# Footer note with Google Sheets integration information
st.markdown(f"""
    <div class="footer-note">
        <strong>📌 Important:</strong> This tool provides AI-based guidance and records user feedback for model improvements. 
        <br><br>
        <strong>📊 Google Sheets Integration:</strong> 
        {'✅ Connected - Feedback is being saved to your Google Sheets.' if stats and stats.get('sheets_status') == 'Connected' else '⚠️ Not Connected - Feedback is being stored locally only.'}
    </div>
""", unsafe_allow_html=True)