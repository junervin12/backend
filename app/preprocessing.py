import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.data.path.append('./nltk_data')
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    return [t for t in tokens if t not in string.punctuation and t not in stop_words]
