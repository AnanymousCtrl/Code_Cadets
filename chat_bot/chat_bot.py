# Import necessary libraries
import nltk
from nltk.chat.util import Chat, reflections
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Define pairs for the chatbot to use in the conversation
pairs = [
    # Greetings
    [
        r"Hi|Hello|Hey",
        ["Hello!", "Hi there!", "Hey!"]
    ],
    # Age
    [
        r"My age is (\d+)|I'm (\d+) years old",
        ["Interesting, I don't have a physical body, so I don't age.",
         "I'm just a program, so I don't have an age."]
    ],
    # Gender
    [
        r"I'm a (boy|girl|man|woman)",
        ["That's great! I'm just a program, so I don't have a gender.",
         "I'm gender-neutral, so I don't identify as male or female."]
    ],
    # Mental health questions
    [
        r"I'm feeling (sad|depressed|anxious|stressed)",
        ["I'm really sorry to hear that. It's important to talk to someone about how you're feeling.",
         "Have you considered reaching out to a mental health professional?",
         "Remember, it's okay to ask for help when you need it."]
    ],
    # General questions
    [
        r"What's your name|Who are you",
        ["I'm a chatbot designed to help analyze mental health based on responses.",
         "I'm here to assist you and provide support."]
    ],
    # Exit
    [
        r"Quit|Exit|Bye",
        ["Goodbye!", "Take care!", "See you later!"]
    ],
    # Mental health diagnosis questions
    [
        r"Do you have trouble sleeping|Do you have trouble concentrating|Do you have thoughts of suicide",
        ["I'm really sorry to hear that. It's important to talk to someone about how you're feeling.",
         "Have you considered reaching out to a mental health professional?",
         "Remember, it's okay to ask for help when you need it."]
    ]
]

# Define the chat function
def chat():
    print("Hi, I'm a chatbot designed to help analyze mental health based on responses.")
    chat = Chat(pairs, reflections)
    chat.converse()

# Define the machine learning model
def train_model(X, y):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    return model, vectorizer

# Define the main function
def main():
    # Load the data
    X = ["I'm feeling sad", "I'm feeling anxious", "I'm feeling depressed", "I'm feeling stressed"]
    y = ["sad", "anxious", "depressed", "stressed"]
    # Train the model
    model, vectorizer = train_model(X, y)
    # Start the chat
    chat()

# Call the main function
if __name__ == "__main__":
    main()