# Fake News Detection

## Overview

This project aims to classify news articles as real or fake using Natural Language Processing (NLP) techniques. The
model is trained on a dataset of labeled fake and real news articles and can predict whether a given news article is
likely to be real or fake. The project involves text preprocessing, feature extraction, binary classification, and model
evaluation.

## Project Objectives

- **Preprocessing News Data**: Clean and preprocess news articles by removing unwanted characters, tokenizing the text,
  and removing stopwords.
- **Vectorization**: Convert text data into numerical features using TF-IDF or word embeddings.
- **Binary Classification**: Train a classification model (Decision Tree) to
  predict the authenticity of news articles.
- **Model Evaluation**: Evaluate the model's performance using accuracy, precision, recall, and F1-score.
- **Deployment**: Deploy the trained model as a REST API to make real-time predictions.

## Dataset

The dataset used in this project consists of two CSV files:

- `Fake.csv`: Contains articles labeled as fake news (class 0).
- `True.csv`: Contains articles labeled as real news (class 1).

Both datasets include columns such as `subject`, `title`, `date`, and `text`. We preprocess the `text` column and use it
for model training.

### Data Preprocessing Steps:

1. **Text Cleaning**: Remove HTML tags, special characters, and punctuation.
2. **Tokenization**: Split the text into words or tokens.
3. **Stopword Removal**: Remove common words (e.g., "the", "and") that don't contribute much to the meaning.
4. **Vectorization**: Convert cleaned text into numerical form using TF-IDF.

## Installation

### Prerequisites:

- Python 3.7+
- Libraries: `pandas`, `scikit-learn`, `joblib`, `nltk`, `numpy`, `h5py`

You can install the required libraries using the following command:

```bash
pip install -r requirements.txt
```

### Create the `requirements.txt` file:

```
pandas
scikit-learn
joblib
nltk
numpy
h5py
```

## Usage

### 1. Preprocessing the Data

To preprocess the dataset and clean the text, use the `preprocess_text.py` script:

```python
from preprocess_text import preprocess_text
import pandas as pd

# Load the datasets
df_fake = pd.read_csv("./assets/Fake.csv")
df_true = pd.read_csv("./assets/True.csv")

df_fake["class"] = 0
df_true["class"] = 1

# Merge the datasets
df_merged = pd.concat([df_fake, df_true], axis=0)
df_merged = df_merged.drop(columns=['subject', 'title', 'date'], axis=1)

# Clean and preprocess the text
X_cleaned = preprocess_text(df_merged['text'], fit_vectorizer=True)

# Save processed data
import joblib

joblib.dump(X_cleaned, "X_features.joblib")
joblib.dump(df_merged['class'], "y_labels.joblib")
```

### 2. Training the Model

After preprocessing the data, you can train the classification model (e.g., Decision Tree):

```python
import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load preprocessed data
X_loaded = joblib.load('X_features.joblib')
y_loaded = joblib.load('y_labels.joblib')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_loaded, y_loaded, test_size=0.2, random_state=42)

# Train a Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

# Save the trained model
joblib.dump(model, 'text_classifier_model.pkl')
```

### 3. Predicting with the Model

Once the model is trained, you can load it and use it to make predictions on new data:

```python
import joblib
import numpy as np
from preprocess_text import preprocess_text

# Load the trained model and TF-IDF vectorizer
model = joblib.load('text_classifier_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Example new text for prediction
new_texts = [
    "Breaking news: The government announces new policies for the economy.",
    "Aliens have landed on Earth and are living among us!"
]

# Preprocess the new texts
X_new = preprocess_text(new_texts, fit_vectorizer=False)

# Predict with the model
predictions = model.predict(X_new)

# Output the predictions
for text, pred in zip(new_texts, predictions):
    print(f"Text: {text}\nPredicted Label: {'Fake News' if pred == 0 else 'Real News'}\n")
```

## Model Performance

During model selection, multiple models were evaluated based on their performance. Below are the results for several models:

### **Logistic Regression**
- **Accuracy**: 98.80%
- **Precision**: 98.53%
- **Recall**: 98.95%
- **F1-Score**: 98.74%

### **Support Vector Classifier (SVC)**
- **Accuracy**: 99.53%
- **Precision**: 99.44%
- **Recall**: 99.58%
- **F1-Score**: 99.51%

### **Gradient Boosting Classifier (GBC)**
- **Accuracy**: 99.54%
- **Precision**: 99.42%
- **Recall**: 99.63%
- **F1-Score**: 99.52%

### **Decision Tree Classifier**
- **Accuracy**: 99.62%
- **Precision**: 99.65%
- **Recall**: 99.56%
- **F1-Score**: 99.60%

After evaluating multiple models, **Decision Tree** was chosen as the best model due to its balance between performance and simplicity, achieving the highest **accuracy**, **precision**, **recall**, and **F1-score** among the tested models.


## Model Evaluation Metrics

- **Accuracy**: The proportion of correctly classified articles.
- **Precision**: The proportion of true positive predictions among all positive predictions.
- **Recall**: The proportion of true positive predictions among all actual positive labels.
- **F1-Score**: The harmonic mean of precision and recall, providing a balance between the two.

## Deployment

You can deploy the trained model as a REST API using a web framework such as Flask or FastAPI. Below is an outline of
how you might set up a simple Flask app to serve the model:

```python
from flask import Flask, request, jsonify
import joblib
from preprocess_text import preprocess_text

app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load('text_classifier_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']

    # Preprocess the input text
    X_new = preprocess_text([text], fit_vectorizer=False)

    # Predict the class
    prediction = model.predict(X_new)

    return jsonify({'prediction': 'Fake News' if prediction[0] == 0 else 'Real News'})


if __name__ == '__main__':
    app.run(debug=True)
```

### Example Request:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"text": "This is a sample news article."}' http://localhost:5000/predict
```

### Example Response:

```json
{
  "prediction": "Real News"
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The dataset used in this project is available
  from [Kaggle](https://www.kaggle.com/code/therealsampat/fake-news-detection).
- This project was built using scikit-learn, pandas, NumPy, and joblib for model training, evaluation, and
  serialization.
