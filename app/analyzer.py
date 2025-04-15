from app.preprocessing import preprocess_text
from app.model import get_model
from nltk.probability import FreqDist
from nltk.util import ngrams
import math
import logging

logging.basicConfig(level=logging.INFO)

def calculate_perplexity(tokens, model):
    from nltk.util import ngrams

    n = model.order
    padded_tokens = ['<s>'] * (n - 1) + tokens + ['</s>']
    ngram_sequence = list(ngrams(padded_tokens, n))

    if not ngram_sequence:
        return -1

    log_prob_sum = 0
    for ngram in ngram_sequence:
        prob = model.score(ngram[-1], ngram[:-1])
        if prob <= 0:
            prob = 1e-10  # fallback probability
        log_prob_sum += -math.log(prob)

    perplexity = math.exp(log_prob_sum / len(ngram_sequence))
    return perplexity


def calculate_burstiness(tokens):
    word_freq = FreqDist(tokens)
    if len(word_freq) == 0:
        return float("nan")
    avg_freq = sum(word_freq.values()) / len(word_freq)
    if avg_freq == 0:
        return float("nan")
    variance = sum((freq - avg_freq) ** 2 for freq in word_freq.values()) / len(word_freq)
    return variance / (avg_freq ** 2)

def analyze_text(text):
    tokens = preprocess_text(text)
    model = get_model()

    perplexity = calculate_perplexity(tokens, model)
    burstiness = calculate_burstiness(tokens)

    # Handle nilai tidak valid
    if not math.isfinite(perplexity):
        logging.warning(f"Perplexity value invalid: {perplexity}")
        perplexity = -1

    if not math.isfinite(burstiness):
        logging.warning(f"Burstiness value invalid: {burstiness}")
        burstiness = -1

    # Menurunkan ambang batas lebih jauh untuk mendeteksi AI-generated text
    verdict = "Likely human-written text"
    if perplexity > 100000 and burstiness > 0.1:  # Menurunkan batas lebih jauh
        verdict = "AI-generated-like text (based on statistical behavior)"

    return {
        "perplexity": perplexity,
        "burstiness": burstiness,
        "verdict": verdict
    }


