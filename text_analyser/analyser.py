import warnings
import pandas as pd
import numpy as np
import spacy
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Suppress the specific FutureWarning from sklearn
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.utils.validation")

# Load spaCy model for Named Entity Recognition
nlp = spacy.load("en_core_web_sm")

# Step 1: Data Collection and Preparation
data = {
    'feeling': [
        "happy", "sad", "anxious", "excited", "depressed", "calm", "stressed", "content",
        "bored", "energetic", "hopeless", "optimistic", "pessimistic", "relaxed", "tense"
    ],
    'sleep_hours': [8, 5, 6, 7, 4, 8, 6, 7, 5, 8, 4, 7, 6, 8, 5],
    'stress_level': [1, 4, 3, 2, 5, 1, 4, 2, 3, 1, 5, 2, 4, 1, 3],
    'mental_health': [1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 for Good, 0 for Not Good
}

df = pd.DataFrame(data)

# Step 2: Feature Engineering
vectorizer = CountVectorizer()
feeling_vectors = vectorizer.fit_transform(df['feeling'])

# Convert all columns to strings
feeling_df = pd.DataFrame(feeling_vectors.toarray(), columns=vectorizer.get_feature_names_out())
feeling_df.columns = feeling_df.columns.astype(str)

# Prepare the final feature set
X = pd.concat([feeling_df, df[['sleep_hours', 'stress_level']].astype(str)], axis=1)
X.columns = X.columns.astype(str)
y = df['mental_health']

# Step 3: Model Selection and Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 4: Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Step 5: Extracting Information from Paragraph
def extract_information(paragraph):
    doc = nlp(paragraph)
    
    feeling = None
    sleep_hours = None
    stress_level = None
    
    feelings_set = set(data['feeling'])  # set of predefined feelings
    
    for token in doc:
        if token.text.lower() in feelings_set:
            feeling = token.text.lower()
        
    sleep_match = re.search(r'(\d+)\s*hours?\s*sleep', paragraph, re.IGNORECASE)
    if sleep_match:
        sleep_hours = int(sleep_match.group(1))
    
    stress_match = re.search(r'(\d+)\s*stress\s*level', paragraph, re.IGNORECASE)
    if stress_match:
        stress_level = int(stress_match.group(1))
    
    return feeling, sleep_hours, stress_level

# Step 6: User Input and Prediction
def predict_mental_health_from_paragraph(paragraph):
    feeling, sleep_hours, stress_level = extract_information(paragraph)
    
    if None in [feeling, sleep_hours, stress_level]:
        return "Could not extract all required information from the paragraph."
    
    feeling_vector = vectorizer.transform([feeling]).toarray()
    input_data = np.concatenate([feeling_vector, [[sleep_hours, stress_level]]], axis=1)
    input_df = pd.DataFrame(input_data, columns=X.columns)
    prediction = model.predict(input_df)
    
    return "Good Mental Health" if prediction[0] == 1 else "Not Good Mental Health"

# Example paragraph input
user_paragraph = "I am feeling very anxious today. I slept for 6 hours last night and my stress level is at 4."
print(f"User Mental Health Prediction: {predict_mental_health_from_paragraph(user_paragraph)}")