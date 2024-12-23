import joblib
from preprocess_text import preprocess_text

# Load the saved model
model = joblib.load("text_classifier_model.pkl")


def make_prediction(texts):
    # Preprocess the new texts (fit_vectorizer=False ensures it uses the existing TF-IDF vectorizer)
    X_new = preprocess_text(texts, fit_vectorizer=False)

    # Predict the class of the new texts
    predictions = model.predict(X_new)

    # Map the predictions to their labels
    label_map = {0: "Fake News", 1: "Real News"}
    predicted_labels = [label_map[pred] for pred in predictions]

    format_pred = []

    # Display the results
    for text, label in zip(texts, predicted_labels):
        format_pred.append({"text": text, "prediction": label})
        # print(f"Text: {text}\nPredicted Label: {label}\n")

    return format_pred


# Example new text for prediction
# new_texts = [
#     "Breaking news: The government announces new policies for the economy.",
#     "Aliens have landed on Earth and are living among us!",
#     "The following statements were posted to the verified Twitter accounts of U.S. President Donald Trump, @realDonaldTrump and @POTUS.  The opinions expressed are his own.Â Reuters has not edited the statements or confirmed their accuracy.  @realDonaldTrump : - Together, we are MAKING AMERICA GREAT AGAIN! bit.ly/2lnpKaq [1814 EST] - In the East, it could be the COLDEST New Yearâ€™s Eve on record."
# ]
#
# result = make_prediction(new_texts)
# print(result)
