import streamlit as st
import joblib
import re
import nltk
import pandas as pd
import requests
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import time

# Download NLTK resources
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load model and vectorizer
@st.cache_resource
def load_model():
    try:
        model = joblib.load("stacking_model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

model, vectorizer = load_model()

# Initialize Porter Stemmer
ps = PorterStemmer()

def preprocess(sentence):
    """Preprocess text by cleaning, lowercasing, removing stopwords and stemming"""
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    sentence = sentence.lower().split()
    sentence = [ps.stem(word) for word in sentence if word not in stopwords.words('english')]
    return ' '.join(sentence)

def predict_fake_news(text):
    """Predict if the given text is fake or real news"""
    if not text.strip():
        return None, "Empty text provided"
    
    try:
        processed = preprocess(text)
        vectorized = vectorizer.transform([processed])
        prediction = model.predict(vectorized)[0]
        confidence = model.predict_proba(vectorized)[0]
        
        # Get confidence score of the prediction
        conf_score = confidence[0] if prediction == 0 else confidence[1]
        
        return prediction, conf_score
    except Exception as e:
        return None, f"Error in prediction: {e}"

def extract_article_from_url(url):
    """Extract article text from a given URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise error for bad responses
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Extract text from paragraphs (most common for articles)
        paragraphs = soup.find_all('p')
        article_text = ' '.join([para.get_text() for para in paragraphs])
        
        if not article_text.strip():
            # If no paragraphs found, get all text
            article_text = soup.get_text(separator=' ', strip=True)
        
        return article_text.strip()
    except Exception as e:
        st.error(f"Error extracting article: {e}")
        return None

# UI Components
def show_header():
    st.title("📰 Fake News Detector")
    st.markdown("""
    This application uses machine learning to analyze news articles and determine if they're likely to be fake news or real news.
    """)
    
def show_example():
    with st.expander("ℹ️ How to use this app"):
        st.markdown("""
        1. **URL Input**: Enter a URL of a news article
        2. **Text Input**: Or paste the full text of an article
        3. **Click 'Analyze'** to get the prediction
        
        The model will analyze the content and return a prediction of whether the article appears to be real or fake news.
        """)

def show_result(prediction, confidence):
    st.markdown("### Analysis Result")
    
    if prediction == 0:
        st.error(f"🔴 This news is likely FAKE (Confidence: {confidence:.2%})")
        st.markdown("""
        **What this means**: The article shows patterns commonly found in misleading or false information.
        
        **Why this might happen**:
        - Sensationalist language
        - Lack of credible sources
        - Extreme claims with little evidence
        - Strong emotional language
        """)
    else:
        st.success(f"🟢 This news is likely REAL (Confidence: {confidence:.2%})")
        st.markdown("""
        **What this means**: The article appears to follow patterns of legitimate news reporting.
        
        **Note**: While the model analyzes language patterns, always:
        - Check the source credibility
        - Look for multiple sources reporting the same story
        - Verify with fact-checking websites
        """)

def show_recent_analysis():
    if 'history' in st.session_state and st.session_state.history:
        st.markdown("### Recent Analysis")
        
        for i, (text, pred, conf) in enumerate(st.session_state.history[-3:]):
            with st.expander(f"Article {len(st.session_state.history) - i}"):
                st.markdown(f"**Preview**: {text[:150]}...")
                if pred == 0:
                    st.markdown(f"**Result**: 🔴 Likely FAKE (Confidence: {conf:.2%})")
                else:
                    st.markdown(f"**Result**: 🟢 Likely REAL (Confidence: {conf:.2%})")

def main():
    show_header()
    show_example()
    
    # Initialize session state for history
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    # Tabs for different input methods
    tab1, tab2 = st.tabs(["📝 Enter Text", "🔗 Enter URL"])
    
    with tab1:
        text_input = st.text_area("Paste article text here", height=200)
        submit_text = st.button("Analyze Text")
        
        if submit_text and text_input:
            with st.spinner("Analyzing text..."):
                prediction, confidence = predict_fake_news(text_input)
                
                if isinstance(confidence, str):  # Error message
                    st.error(confidence)
                else:
                    show_result(prediction, confidence)
                    # Save to history
                    st.session_state.history.append((text_input, prediction, confidence))
    
    with tab2:
        url_input = st.text_input("Enter article URL", placeholder="https://example.com/news-article")
        submit_url = st.button("Analyze URL")
        
        if submit_url and url_input:
            with st.spinner("Fetching and analyzing article..."):
                article_text = extract_article_from_url(url_input)
                
                if article_text:
                    st.markdown("### Article Preview")
                    st.text_area("Extracted text", article_text[:500] + "...", height=150, disabled=True)
                    
                    prediction, confidence = predict_fake_news(article_text)
                    
                    if isinstance(confidence, str):  # Error message
                        st.error(confidence)
                    else:
                        show_result(prediction, confidence)
                        # Save to history
                        st.session_state.history.append((article_text, prediction, confidence))
    
    # Show recent analysis history
    show_recent_analysis()
    
    # Footer
    st.markdown("---")
    st.markdown("Note: This tool provides an AI-based estimation and should not be the sole determinant of an article's credibility.")

if __name__ == "__main__":
    main()