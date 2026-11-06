"""Cleaners for English text normalization (including time, date/year, abbreviations, and numbers)."""

import re
import unicodedata
try:
    from unidecode import unidecode
except ImportError:
    def unidecode(s):
        return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
from .numbers import normalize_numbers

_whitespace_re = re.compile(r"\s+")
_time_am_pm_re = re.compile(r"\b(\d{1,2}):(\d{2})\s*([aApP][mM])\b")
_time_24h_re = re.compile(r"\b(\d{1,2}):(\d{2})\b")

_abbreviations = [
    (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
    for x in [
        ("mrs", "misess"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
    ]
]


def _expand_time_am_pm(match):
    hour = int(match.group(1))
    minute = int(match.group(2))
    meridiem = match.group(3).lower()
    meridiem_spoken = " ".join(list(meridiem))  # "a m" or "p m"
    if minute == 0:
        return f"{hour} {meridiem_spoken}"
    elif minute < 10:
        return f"{hour} oh {minute} {meridiem_spoken}"
    else:
        return f"{hour} {minute} {meridiem_spoken}"


def _expand_time_24h(match):
    hour = int(match.group(1))
    minute = int(match.group(2))
    if minute == 0:
        return f"{hour} hundred"
    elif minute < 10:
        return f"{hour} oh {minute}"
    else:
        return f"{hour} {minute}"


def expand_times(text):
    text = re.sub(_time_am_pm_re, _expand_time_am_pm, text)
    text = re.sub(_time_24h_re, _expand_time_24h, text)
    return text


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def expand_numbers(text):
    return normalize_numbers(text)


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)


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
    """Pipeline for English text, including time, number, and abbreviation expansion."""
    text = convert_to_ascii(text)
    text = expand_times(text)
    text = lowercase(text)
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = collapse_whitespace(text)
    return text
