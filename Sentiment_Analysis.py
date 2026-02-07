import pandas as pd
import nltk
import string

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load dataset
data = pd.read_csv("dataset.csv")

# Text preprocessing function
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return " ".join(words)

# Apply preprocessing
data['text'] = data['text'].apply(preprocess)

# Feature extraction
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['text'])
y = data['sentiment']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# User input testing
while True:
    text = input("\nEnter text for sentiment analysis (or type exit): ")
    if text.lower() == "exit":
        break
    processed = preprocess(text)
    vector = vectorizer.transform([processed])
    result = model.predict(vector)
    print("Sentiment:", result[0])