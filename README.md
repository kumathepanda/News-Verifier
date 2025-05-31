# 📰 News Verifier

An AI-powered web application built with Streamlit that helps users verify the authenticity of news content using machine learning models.

## 🌟 Features

- **AI-Powered Analysis**: Uses trained machine learning models to classify news content as authentic or fake
- **Real-time Predictions**: Provides confidence scores for both fake and authentic classifications
- **User Feedback System**: Collects user feedback to continuously improve model accuracy
- **Google Sheets Integration**: Automatically saves feedback data to Google Sheets for analysis
- **Beautiful UI**: Coffee-themed, newspaper-inspired design with smooth animations
- **Session Management**: Tracks user interactions and feedback within sessions
- **Data Persistence**: Supports both cloud (Google Sheets) and local data storage

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- pip (Python package manager)
- Google Cloud Service Account (optional, for Google Sheets integration)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/kumathepanda/News-Verifier.git
   cd news-verifier
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up required model files**
   
   Ensure you have these files in your project directory:
   - `fake_news_model.pkl` - Your trained ML model
   - `tfidf_vectorizer.pkl` - Your TF-IDF vectorizer

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   
   Navigate to `http://localhost:8501` to access the application.

## 📦 Dependencies

```txt
streamlit>=1.28.0
joblib>=1.3.0
nltk>=3.8.0
pandas>=2.0.0
gspread>=5.10.0
google-auth>=2.20.0
```

Create a `requirements.txt` file with the above dependencies.

## ⚙️ Configuration

### Google Sheets Integration (Optional)

To enable automatic feedback saving to Google Sheets:

1. **Create a Google Cloud Project**
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select existing one

2. **Enable Google Sheets API**
   - Navigate to APIs & Services > Library
   - Search for "Google Sheets API" and enable it
   - Also enable "Google Drive API"

3. **Create Service Account**
   - Go to APIs & Services > Credentials
   - Click "Create Credentials" > "Service Account"
   - Download the JSON key file

4. **Configure Streamlit Secrets**
   
   Create `.streamlit/secrets.toml`:
   ```toml
   [gcp_service_account]
   type = "service_account"
   project_id = "your-project-id"
   private_key_id = "your-private-key-id"
   private_key = "-----BEGIN PRIVATE KEY-----\nYour-Private-Key\n-----END PRIVATE KEY-----\n"
   client_email = "your-service-account@your-project.iam.gserviceaccount.com"
   client_id = "your-client-id"
   auth_uri = "https://accounts.google.com/o/oauth2/auth"
   token_uri = "https://oauth2.googleapis.com/token"
   auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
   client_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
   ```

5. **Update Configuration**
   
   In `app.py`, update the Google Sheets URL:
   ```python
   GOOGLE_SHEETS_URL = "https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID/edit"
   ```

### Alternative: Local Development

For local development without Google Sheets:
- Place your service account JSON file as `service_account.json` in the project root
- The app will automatically use local fallback storage

## 🏗️ Project Structure

```
news-verifier/
├── app.py                          # Main Streamlit application
├── fake_news_model.pkl             # Trained ML model (joblib format)
├── tfidf_vectorizer.pkl            # TF-IDF vectorizer (joblib format)
├── requirements.txt                # Python dependencies
├── service_account.json            # Google Service Account (optional)
├── feedback_data_backup.csv        # Local backup of feedback data
├── .streamlit/
│   └── secrets.toml               # Streamlit secrets configuration
└── README.md                      # This file
```

## 🔧 Usage

### Basic Usage

1. **Enter News Content**: Paste any news article, headline, or text content into the text area
2. **Click Analyze**: Press the "🔍 Analyze Content" button
3. **View Results**: See the prediction with confidence scores
4. **Provide Feedback** (Optional): Help improve the model by correcting wrong predictions

### Feedback System

The app includes a human-in-the-loop feedback system:
- Users can correct model predictions
- Feedback is stored for model retraining
- Statistics show community contributions
- Data is saved to Google Sheets (if configured) or locally

## 🧠 Model Information

The application expects:
- **Model Format**: scikit-learn compatible model saved with joblib
- **Vectorizer**: TF-IDF vectorizer that transforms text to numerical features
- **Labels**: Binary classification (0 = Fake, 1 = Real)

### Text Preprocessing

The app applies the following preprocessing:
- Lowercase conversion
- URL and HTML tag removal
- Non-alphabetic character removal
- Tokenization using NLTK
- Stop word removal
- Lemmatization
- Minimum word length filtering (>2 characters)

## 📊 Data Collection

Feedback data includes:
- `clean_text`: Preprocessed text content
- `label`: Corrected label (0 or 1)
- `timestamp`: When feedback was provided
- `session_id`: Unique session identifier

## 🎨 UI Features

- **Responsive Design**: Works on desktop and mobile devices
- **Coffee Theme**: Warm, newspaper-inspired color scheme
- **Smooth Animations**: Hover effects and transitions
- **Progress Bars**: Visual confidence indicators
- **Toast Notifications**: Success messages for user actions

## 🔒 Privacy & Security

- **No Personal Data**: Only news content and predictions are stored
- **Session-based**: No persistent user tracking
- **Local Fallback**: Works without internet connectivity for core features
- **Secure API**: Google Sheets integration uses service account authentication

## 🐛 Troubleshooting

### Common Issues

1. **NLTK Data Download Errors**
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
   ```

2. **Model Loading Errors**
   - Ensure `fake_news_model.pkl` and `tfidf_vectorizer.pkl` are in the project directory
   - Verify models were saved with compatible joblib/scikit-learn versions

3. **Google Sheets Connection Issues**
   - Check service account permissions
   - Ensure the Google Sheet is shared with the service account email
   - Verify API is enabled in Google Cloud Console

4. **Streamlit Caching Issues**
   ```bash
   streamlit cache clear
   ```

## 📈 Deployment

### Local Development
```bash
streamlit run app.py
```

### Streamlit Cloud
1. Push code to Git repository
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Add secrets in the Streamlit Cloud dashboard
4. Deploy directly from GitHub

### Docker (Optional)
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -am 'Add some feature'`
4. Push to branch: `git push origin feature/your-feature`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Natural Language Processing with [NLTK](https://www.nltk.org/)
- Machine Learning with [scikit-learn](https://scikit-learn.org/)
- Data storage with [Google Sheets API](https://developers.google.com/sheets/api)

## 📞 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section above
2. Search existing issues in the repository
3. Create a new issue with detailed information
4. Include error messages and steps to reproduce

---

**Happy News Verification! 📰✨**
