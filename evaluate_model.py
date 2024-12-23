import pandas as pd
from nltk.classify.svm import SvmClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBRegressor

from preprocess_text import preprocess_text

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

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y, test_size=0.2, random_state=42)

def evaluate_models():
    # Initialize models
    models = {
        "Logistic Regression": LogisticRegression(),
        "SVC": SVC(kernel='linear'),
        "GBC": GradientBoostingClassifier(random_state=0),
        "Decision Tree": DecisionTreeClassifier(),
    }

    # Perform training, evaluation, and cross-validation
    for name, model in models.items():
        print(f"Model: {name}")

        # Train the model
        model.fit(X_train, y_train)

        # Evaluate on test set
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1-Score: {f1}")
        print("-" * 50)


# evaluate_models()
