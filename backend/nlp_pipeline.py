"""
NLP Preprocessing Pipeline for AI Resume Screening & ATS Score Predictor.
Handles text normalization, cleaning, tokenization, lemmatization, stopword removal,
and n-gram generation.
"""

import re
import string
from typing import List, Set

# Attempt to import NLTK with automated quiet resource acquisition
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    try:
        nltk.data.find("tokenizers/punkt")
    except (LookupError, AttributeError):
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)

    try:
        nltk.data.find("corpora/stopwords")
    except (LookupError, AttributeError):
        nltk.download("stopwords", quiet=True)

    try:
        nltk.data.find("corpora/wordnet")
    except (LookupError, AttributeError):
        nltk.download("wordnet", quiet=True)

    NLTK_AVAILABLE = True
    STOP_WORDS: Set[str] = set(stopwords.words("english"))
    LEMMATIZER = WordNetLemmatizer()
except Exception:
    NLTK_AVAILABLE = False
    # Standard English stop words fallback
    STOP_WORDS = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
        "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
        "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
        "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
        "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
        "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
        "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
        "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
        "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
        "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
        "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
        "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
        "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
        "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
        "they've", "this", "those", "through", "to", "too", "under", "until", "up",
        "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
        "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
        "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
        "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
        "yourself", "yourselves"
    }
    LEMMATIZER = None

# Custom domain stopwords to filter boilerplate resume tokens
DOMAIN_NOISE_WORDS = {
    "resume", "curriculum", "vitae", "cv", "page", "phone", "email",
    "address", "contact", "references", "available", "upon", "request",
    "date", "birth", "gender", "status", "declaration", "hereby"
}
COMBINED_STOPWORDS = STOP_WORDS.union(DOMAIN_NOISE_WORDS)


class NLPPipeline:
    """
    Modular NLP Preprocessing Pipeline providing text cleaning,
    tokenization, stopword elimination, lemmatization, and phrase extraction.
    """

    def __init__(self):
        self.stop_words = COMBINED_STOPWORDS
        self.lemmatizer = LEMMATIZER

    def clean_text(self, text: str, preserve_case: bool = False) -> str:
        """
        Cleans and sanitizes raw input text:
        - Removes URLs, web links, and file paths
        - Normalizes non-ASCII characters & whitespace
        - Preserves technical symbols (like C++, C#, .NET)
        """
        if not text:
            return ""

        # Remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)

        # Normalize linebreaks and tabs
        text = re.sub(r"[\r\n\t]+", " ", text)

        # Normalize special unicode quotes/dashes
        text = text.replace("’", "'").replace("“", '"').replace("”", '"').replace("–", "-")

        # Keep characters relevant to tech skills: e.g. C++, C#, .NET, Node.js
        # Replace non-alphanumeric except specific tech symbols (+, #, ., -, /)
        text = re.sub(r"[^\w\s\+\#\.\-/]", " ", text)

        # Collapse repeated spaces
        text = re.sub(r"\s+", " ", text).strip()

        if not preserve_case:
            text = text.lower()

        return text

    def tokenize(self, text: str) -> List[str]:
        """Tokenizes sanitized text into individual words/tokens."""
        if not text:
            return []

        cleaned = self.clean_text(text, preserve_case=False)

        if NLTK_AVAILABLE:
            try:
                tokens = word_tokenize(cleaned)
            except Exception:
                tokens = re.findall(r"\b[\w\+\#\.\-]+\b", cleaned)
        else:
            tokens = re.findall(r"\b[\w\+\#\.\-]+\b", cleaned)

        return tokens

    def lemmatize_token(self, token: str) -> str:
        """Lemmatizes a single token to its base dictionary form."""
        if self.lemmatizer:
            try:
                # First try verb, then noun lemmatization
                lemma = self.lemmatizer.lemmatize(token, pos="v")
                return self.lemmatizer.lemmatize(lemma, pos="n")
            except Exception:
                return token
        return token

    def preprocess_tokens(
        self,
        text: str,
        remove_stops: bool = True,
        lemmatize: bool = True
    ) -> List[str]:
        """
        End-to-end token preprocessing:
        Cleans -> Tokenizes -> Filters Stopwords -> Lemmatizes.
        """
        raw_tokens = self.tokenize(text)
        processed: List[str] = []

        for token in raw_tokens:
            tok = token.strip().lower()
            if len(tok) <= 1 and tok not in {"c", "r"}:
                continue
            if remove_stops and tok in self.stop_words:
                continue

            if lemmatize:
                tok = self.lemmatize_token(tok)

            processed.append(tok)

        return processed

    def get_ngrams(self, tokens: List[str], n: int = 2) -> List[str]:
        """Generates n-gram tuples joined by single space."""
        if len(tokens) < n:
            return []
        return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    def extract_candidate_keywords(self, text: str) -> Set[str]:
        """
        Extracts uni-, bi-, and tri-grams from text to capture multi-word
        skill phrases such as 'machine learning' or 'cloud computing'.
        """
        tokens = self.preprocess_tokens(text, remove_stops=True, lemmatize=False)
        unigrams = set(tokens)
        bigrams = set(self.get_ngrams(tokens, 2))
        trigrams = set(self.get_ngrams(tokens, 3))
        return unigrams.union(bigrams).union(trigrams)


# Shared global pipeline instance
nlp_pipeline = NLPPipeline()
