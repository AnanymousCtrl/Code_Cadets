import pandas as pd
data = pd.read_csv('D:\\Codes\\Python For DS\\Kriyeta App\\app2\\datam.csv')
print(data.head())

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(Text):
    
    tokens = word_tokenize(Text.lower())
    tokens = [word for word in tokens if word.isalnum() and word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

data['processed_text'] = data['Text'].apply(preprocess_text)
print(data.head())

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['processed_text'])
y = data['Label'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))

def predict_mental_health(Text):
    processed_text = preprocess_text(Text)
    vectorized_text = vectorizer.transform([processed_text])
    prediction = model.predict(vectorized_text)
    return prediction[0]

new_text = "i am not good in mood any more now"
prediction = predict_mental_health(new_text)
print('Prediction:', predict_mental_health(new_text))

if (prediction == 1):
    print("Depression and Anxiety")
else:
    print("Good Mental Health")