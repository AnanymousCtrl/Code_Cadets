import pandas as pd
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
    negation = False
    processed_tokens = []
    for token in tokens:
        if token in ["not", "no"]:
            negation = True
        elif negation:
            processed_tokens.append("not_" + token)
            negation = True
        else:
            if token.isalnum() and token not in stopwords.words('english'):
                processed_tokens.append(token)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in processed_tokens]
    return ' '.join(tokens)

data = pd.read_csv('TEST_Codes\datam.csv')
data['processed_text'] = data['Text'].apply(preprocess_text)

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['processed_text'])
y = data['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

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

new_text = "I am not feeling happy"
prediction = predict_mental_health(new_text)
print('Prediction:', predict_mental_health(new_text))

if (prediction == 0):
    print("Good mental health")
elif(prediction == 1):
    print("Depression and Anxiety")
elif(prediction == 2):
    print("You are in good mood do something that makes you happy!!")
else:
    print("Let me connect you with someone to help you out!!")