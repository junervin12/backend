from app.preprocessing import preprocess_text
from app.model import get_model
from nltk.probability import FreqDist
import math
import logging

logging.basicConfig(level=logging.INFO)

def calculate_perplexity(tokens, model):
    from nltk.util import ngrams
    padded_tokens = ['<s>'] + tokens + ['</s>']
    ngram_sequence = list(ngrams(padded_tokens, model.order))
    try:
        perplexity = model.perplexity(ngram_sequence)
    except Exception as e:
        logging.warning(f"Failed to calculate perplexity: {e}")
        perplexity = float("inf")
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

    verdict = "Likely human-written text"
    if perplexity < 100 and burstiness < 1:
        verdict = "AI-generated-like text (based on statistical behavior)"

    return {
        "perplexity": perplexity,
        "burstiness": burstiness,
        "verdict": verdict
    }
