"""Cleaners for Telugu and English text."""

import re
import unicodedata
try:
    from unidecode import unidecode
except ImportError:
    def unidecode(s):
        return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
from .numbers import normalize_numbers
from .telugu_numbers import normalize_telugu_numbers

_whitespace_re = re.compile(r"\s+")


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text).strip()


def convert_to_ascii(text):
    return unidecode(text)


def basic_cleaners(text):
    """Basic pipeline that lowercases and collapses whitespace without transliteration."""
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def transliteration_cleaners(text):
    """Pipeline for non-English text that transliterates to ASCII."""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def english_cleaners(text):
    """Pipeline for English text."""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = normalize_numbers(text)
    text = collapse_whitespace(text)
    return text


def telugu_cleaners(text):
    """
    Pipeline for Telugu text:
    1. Unicode NFC normalization
    2. Expands times (10:45), ordinals (12వ), and numbers (2026) into Telugu words
    3. Collapses multiple spaces while preserving deliberate word boundaries (e.g., 'అడ విలో')
    """
    text = unicodedata.normalize("NFC", text)
    text = normalize_telugu_numbers(text)
    text = collapse_whitespace(text)
    return text
