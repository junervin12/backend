from nltk.corpus import brown
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import WittenBellInterpolated

def get_model():
    corpus = brown.sents(categories='news')  # atau bisa semua: brown.sents()
    n = 2
    train_data, vocab = padded_everygram_pipeline(n, corpus)
    model = WittenBellInterpolated(n)
    model.fit(train_data, vocab)
    model.order = n
    return model
