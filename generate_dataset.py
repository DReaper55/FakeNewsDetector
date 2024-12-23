import joblib
import pandas as pd
from preprocess_text import preprocess_text
import numpy as np

df_fake = pd.read_csv("./assets/Fake.csv")
df_true = pd.read_csv("./assets/True.csv")

df_fake["class"] = 0
df_true["class"] = 1

df_merged = pd.concat([df_fake, df_true], axis=0)

columns_to_drop = ['subject', 'title', 'date']

df = df_merged.drop(columns_to_drop, axis=1)
df = df.sample(frac = 1)

X = df['text']
y = df['class']

X_cleaned = preprocess_text(X, fit_vectorizer=True)

X_array = np.array(X_cleaned)
y_array = np.array(y)

# Save the data using joblib
joblib.dump(X_array, 'assets/X_features.joblib')
joblib.dump(y_array, 'assets/y_labels.joblib')

print("Data saved to X_features.joblib and y_labels.joblib")

