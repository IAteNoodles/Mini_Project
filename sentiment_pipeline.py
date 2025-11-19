"""sentiment_pipeline.py

Simple NLP pipeline for sentiment analysis using NLTK movie_reviews dataset.
- Downloads NLTK data if needed
- Loads movie_reviews corpus
- Preprocesses text (lowercasing, basic token cleanup)
- Splits into train/test (80/20nltk.download('words'))
- Builds an sklearn Pipeline: CountVectorizer -> MultinomialNB
- Trains, predicts, and prints accuracy, classification report, and confusion matrix
"""

import nltk
from nltk.corpus import movie_reviews
import random
import re
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def download_nltk_resources():
    try:
        nltk.data.find('corpora/movie_reviews')
    except LookupError:
        print('Downloading NLTK movie_reviews corpus...')
        nltk.download('movie_reviews')


def load_movie_reviews():
    # movie_reviews.fileids() returns filenames like 'neg/cv000_29416.txt'
    texts = []
    labels = []
    for fileid in movie_reviews.fileids():
        label = movie_reviews.categories(fileid)[0]
        raw = movie_reviews.raw(fileid)
        texts.append(raw)
        labels.append(label)
    return texts, labels


def simple_preprocess(text):
    # Lowercase, remove non-alphanumeric characters (keep spaces), collapse whitespace
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


if __name__ == '__main__':
    download_nltk_resources()

    texts, labels = load_movie_reviews()

    # Shuffle dataset to ensure random distribution
    combined = list(zip(texts, labels))
    random.seed(42)
    random.shuffle(combined)
    texts, labels = zip(*combined)

    # Preprocess texts
    texts = [simple_preprocess(t) for t in texts]

    # Train-test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Build the pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('clf', MultinomialNB()),
    ])

    print('Training the pipeline...')
    pipeline.fit(X_train, y_train)

    print('Predicting on test set...')
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {acc:.4f}')

    print('\nClassification Report:')
    print(classification_report(y_test, y_pred, digits=4))

    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
