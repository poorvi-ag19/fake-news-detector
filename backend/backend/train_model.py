import pandas as pd
import re
import nltk
import joblib
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Download NLTK dependencies
nltk.download("punkt")
nltk.download("wordnet")

# Load Fake and Real News JSON files
fake_news = pd.read_json("backend/fake.json")
real_news = pd.read_json("backend/true.json")

# Add labels (0 = Fake, 1 = Real)
fake_news["label"] = 0
real_news["label"] = 1

# Combine both datasets
df = pd.concat([fake_news, real_news], ignore_index=True)

# Function to clean text
def clean_text(text):
    text = re.sub(r"\W", " ", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text)  # Remove extra spaces
    text = text.lower().strip()  # Convert to lowercase
    return text

df["text"] = df["text"].apply(clean_text)

# Tokenization and Lemmatization
lemmatizer = WordNetLemmatizer()
def lemmatize_text(text):
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(lemmatized_tokens)

df["text"] = df["text"].apply(lemmatize_text)

# Define features (X) and labels (y)
X = df["text"]
y = df["label"]

# Convert text to numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(X).toarray()

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train Passive Aggressive Classifier
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, "fake_news_detector.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("✅ Model and vectorizer saved successfully!")
