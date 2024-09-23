import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pickle

# Load the dataset
df = pd.read_csv(r"C:\Users\akash\Downloads\Fake-News-Detection-App-main\Fake-News-Detection-App-main\news.csv")

# Ensure the dataset has the necessary columns
if 'text' not in df.columns or 'label' not in df.columns:
    raise ValueError("Dataset must contain 'text' and 'label' columns.")

# Prepare features and target variable
X = df['text']  # Features (news content)
y = df['label']  # Target variable (fake/real)

# Vectorization
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Save the vectorizer and model
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('finalized_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model and vectorizer have been saved successfully!")
