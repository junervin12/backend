from nltk.corpus import brown
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE

_model = None

def get_model():
    global _model
    if _model is None:
        train_tokens = brown.words()
        train_data, padded_vocab = padded_everygram_pipeline(1, train_tokens)
        model = MLE(1)
        model.fit(train_data, padded_vocab)
        _model = model
    return _model
