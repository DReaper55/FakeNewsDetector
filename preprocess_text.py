import re

import joblib
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('stopwords')

# Predefined set of stopwords
stop_words = set(stopwords.words('english'))

def preprocess_text(texts, fit_vectorizer=False):
    """
    Preprocesses a list of text documents by cleaning, tokenizing, removing stopwords, and vectorizing.

    Args:
        texts (list of str): List of text documents to preprocess.
        fit_vectorizer (bool): If True, fits the TF-IDF vectorizer on the input texts.
                               Use True during training and False during inference.

    Returns:
        vectorized_texts: TF-IDF vectorized representation of the input texts.
    """
    def clean_and_tokenize(text):
        # Clean the text (remove special characters, punctuation, convert to lowercase)
        text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = text.lower()  # Convert to lowercase

        # Tokenize the text
        tokens = word_tokenize(text)

        # Remove stopwords
        filtered_tokens = [word for word in tokens if word not in stop_words]

        # Return the cleaned, tokenized, and filtered text as a single string
        return ' '.join(filtered_tokens)

    # Apply cleaning and tokenization to each text document
    cleaned_texts = [clean_and_tokenize(text) for text in texts]

    # Vectorize the cleaned texts
    if fit_vectorizer:
        tfidf_vectorizer = TfidfVectorizer()
        vectorized_texts = tfidf_vectorizer.fit_transform(cleaned_texts)  # Fit and transform during training

        # Save the fitted TF-IDF vectorizer
        joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.pkl")
    else:
        tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
        vectorized_texts = tfidf_vectorizer.transform(cleaned_texts)  # Only transform during inference

    return vectorized_texts
