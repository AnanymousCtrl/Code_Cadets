from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


data = pd.read_csv('static/datam.csv')

def preprocess_text(Text):
    tokens = word_tokenize(Text.lower())
    tokens = [word for word in tokens if word.isalnum() and word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

data['processed_text'] = data['Text'].apply(preprocess_text)

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['processed_text'])
y = data['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

app = Flask(__name__)

phq9_questions = [
    "Little interest or pleasure in doing things.",
    "Feeling down, depressed, or hopeless.",
    "Trouble falling or staying asleep, or sleeping too much.",
    "Feeling tired or having little energy.",
    "Poor appetite or overeating.",
    "Feeling bad about yourself – or that you are a failure or have let yourself or your family down.",
    "Trouble concentrating on things, such as reading the newspaper or watching television.",
    "Moving or speaking so slowly that other people could have noticed. Or the opposite – being so fidgety or restless that you have been moving around a lot more than usual.",
    "Thoughts that you would be better off dead, or of hurting yourself in some way."
]

gad7_questions = [
    "Feeling nervous, anxious, or on edge.",
    "Not being able to stop or control worrying.",
    "Worrying too much about different things.",
    "Trouble relaxing.",
    "Being so restless that it’s hard to sit still.",
    "Becoming easily annoyed or irritable.",
    "Feeling afraid as if something awful might happen."
]

phq9_recommendations = {
    range(0, 5): "Minimal depression. Suggestions: Maintain a healthy lifestyle, engage in social activities.",
    range(5, 10): "Mild depression. Suggestions: Consider therapy, practice mindfulness.",
    range(10, 15): "Moderate depression. Suggestions: Therapy is recommended, possibly medication.",
    range(15, 20): "Moderately severe depression. Suggestions: Seek medical advice, medication likely needed.",
    range(20, 28): "Severe depression. Suggestions: Immediate medical attention required, intensive treatment."
}

gad7_recommendations = {
    range(0, 5): "Minimal anxiety. Suggestions: Maintain a healthy lifestyle, practice relaxation techniques.",
    range(5, 10): "Mild anxiety. Suggestions: Consider therapy, engage in stress-reducing activities.",
    range(10, 15): "Moderate anxiety. Suggestions: Therapy is recommended, consider medication.",
    range(15, 21): "Severe anxiety. Suggestions: Seek medical advice, likely need for medication."
}

def get_recommendation(score, recommendations):
    for score_range, recommendation in recommendations.items():
        if score in score_range:
            return recommendation
    return "No recommendation available."

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/phq9', methods=['GET', 'POST'])
def phq9():
    if request.method == 'POST':
        scores = [int(request.form[f'question_{i}']) for i in range(len(phq9_questions))]
        total_score = sum(scores)
        recommendation = get_recommendation(total_score, phq9_recommendations)
        return redirect(url_for('results', test='PHQ-9', score=total_score, recommendation=recommendation))
    return render_template('phq9.html', questions=phq9_questions, enumerate=enumerate)

@app.route('/gad7', methods=['GET', 'POST'])
def gad7():
    if request.method == 'POST':
        scores = [int(request.form[f'question_{i}']) for i in range(len(gad7_questions))]
        total_score = sum(scores)
        recommendation = get_recommendation(total_score, gad7_recommendations)
        return redirect(url_for('results', test='GAD-7', score=total_score, recommendation=recommendation))
    return render_template('gad7.html', questions=gad7_questions, enumerate=enumerate)

@app.route('/results')
def results():
    test = request.args.get('test')
    score = request.args.get('score')
    recommendation = request.args.get('recommendation')
    return render_template('results.html', test=test, score=score, recommendation=recommendation)

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['text']
    processed_text = preprocess_text(user_input)
    vectorized_text = vectorizer.transform([processed_text])
    prediction = model.predict(vectorized_text)
    result = "Good Mental Health" if prediction[0] == 0 else "Depression and Anxiety"
    return render_template('home.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
