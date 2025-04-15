from nltk.corpus import brown
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import WittenBellInterpolated

_model = None

def get_model():
    global _model
    if _model is None:
        train_tokens = brown.words()
        train_data, padded_vocab = padded_everygram_pipeline(2, train_tokens)  # Bigram
        model = WittenBellInterpolated(2)  # Smoothing!
        model.fit(train_data, padded_vocab)
        _model = model
    return _model
