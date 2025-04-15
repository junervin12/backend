from app.preprocessing import preprocess_text
from app.model import get_model
from nltk.probability import FreqDist

def calculate_perplexity(tokens, model):
    from nltk.util import ngrams
    padded_tokens = ['<s>'] + tokens + ['</s>']
    ngram_sequence = list(ngrams(padded_tokens, model.order))
    return model.perplexity(ngram_sequence)

def calculate_burstiness(tokens):
    word_freq = FreqDist(tokens)
    avg_freq = sum(word_freq.values()) / len(word_freq)
    variance = sum((freq - avg_freq) ** 2 for freq in word_freq.values()) / len(word_freq)
    return variance / (avg_freq ** 2)

def analyze_text(text):
    tokens = preprocess_text(text)
    model = get_model()
    perplexity = calculate_perplexity(tokens, model)
    burstiness = calculate_burstiness(tokens)

    verdict = "Likely human-written text"
    if perplexity < 100 and burstiness < 1:
        verdict = "AI-generated-like text (based on statistical behavior)"

    return {
        "perplexity": perplexity,
        "burstiness": burstiness,
        "verdict": verdict
    }
