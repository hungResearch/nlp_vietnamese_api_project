import argparse
import numpy as np
import re
import nltk
from typing import List
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Making the dataset.'
    )
    parser.add_argument(
        '--input_sentence',
        type=str,
        required=True,
        help='A text',
    )
    parser.add_argument(
        '--model_file',
        type=str,
        required=True,
        help='Model',
    )
    return parser.parse_args()


def text_preprocessing(sentence: str) -> str:
    stemmer = WordNetLemmatizer()

    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(sentence))

    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)

    # Converting to Lowercase
    document = document.lower()

    # Lemmatization
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)

    return document


def main():
    args = parse_arguments()
    sentence = args.input_sentence
    clean_sentence = [text_preprocessing(sentence)]
    tfidfconverter = pickle.load(open("feature.pkl", "rb"))
    X = tfidfconverter.transform(clean_sentence).toarray()
    
    with open(args.model_file, 'rb') as training_model:
        model = pickle.load(training_model)
        y_pred2 = model.predict(X)
        print(y_pred2)


if __name__ == "__main__":
    main()
