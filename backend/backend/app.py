from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
nltk.download("punkt")
nltk.download("wordnet")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load trained model and vectorizer
model = joblib.load("fake_news_detector.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Fake news keywords
fake_keywords = ["click to get", "exclusive offer", "win a prize", "you won't believe", 
                 "shocking news", "must see", "limited time", "act now", "secret revealed", "make money fast"]

# Preprocessing function
lemmatizer = WordNetLemmatizer()
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(lemmatized_tokens)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "").lower()  # Lowercase conversion for accurate matching

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Directly return FAKE if text contains spam keywords
    if any(keyword in text for keyword in fake_keywords):
        return jsonify({"label": "FAKE"})

    # Preprocess text and make prediction
    processed_text = preprocess_text(text)
    text_vectorized = vectorizer.transform([processed_text]).toarray()
    prediction = model.predict(text_vectorized)[0]

    result = "REAL" if prediction == 1 else "FAKE"
    return jsonify({"label": result})

if __name__ == "__main__":
    app.run(debug=True)
