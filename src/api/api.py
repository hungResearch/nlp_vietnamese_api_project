from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import re
import nltk
from typing import List
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import logging


app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

class TextRequest(BaseModel):
    text: str

@app.post("/process_text")
def process_text(request: TextRequest):
    try:
        input_text = request.text
        clean_text = [text_preprocessing(input_text)]
        tfidfconverter = pickle.load(open("feature.pkl", "rb"))
        X = tfidfconverter.transform(clean_text).toarray()
        with open("models/model.pkl", 'rb') as training_model:
            model = pickle.load(training_model)
            y_pred2 = model.predict(X)
        if y_pred2[0] == '1':
            return {"sentiment": "Positive"}
        return {"sentiment": "Negative"}

    except Exception as e:
        logger.error(f"Error processing text: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing text: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
