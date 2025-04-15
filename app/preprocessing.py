import nltk
import string
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'nltk_data_new'))
nltk.data.path.append(nltk_data_path)
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    return [t for t in tokens if t not in string.punctuation and t not in stop_words]
