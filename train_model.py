import joblib
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# from lime.lime_text import LimeTextExplainer
import shap

X_loaded = joblib.load('assets/X_features.joblib')
y_loaded = joblib.load('assets/y_labels.joblib')

# Ensure X_loaded is the sparse matrix itself
if isinstance(X_loaded, np.ndarray) and X_loaded.size == 1:
    X_loaded = X_loaded.item()

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_loaded, y_loaded, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()

# Train the model
model.fit(X_train, y_train)

# Evaluate on test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")

# Save the entire model (architecture and weights)
# joblib.dump(model, "text_classifier_model.pkl")

# explainer = LimeTextExplainer(class_names=['real', 'fake'])
# explanation = explainer.explain_instance(text, model.predict_proba)
# explanation.show_in_notebook()

explainer = shap.KernelExplainer(model.predict, X_train)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
