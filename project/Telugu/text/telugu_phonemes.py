"""
Telugu Unicode Grapheme-to-Phoneme (Akshara-to-Phoneme) Module for FastSpeech 2.

Decomposes Telugu script (U+0C00..U+0C7F) into an explicit, unambiguous phone inventory:
- Independent vowels (అ..ఔ) -> vowel phones
- Consonants (క..హ, ళ, ఱ) -> consonant phone + inherent vowel 'a' (unless followed by a dependent vowel matra or virama/halant '్')
- Dependent vowel signs / matras (ా..ౌ) -> replace inherent 'a' with the target vowel phone
- Virama / Halant (్) -> suppresses inherent 'a' to model consonant clusters (conjuncts) and geminates accurately
- Anusvara (ం -> M), Visarga (ః -> H), Candrabindu (ఁ -> N)
"""

import re
import unicodedata
from .telugu_numbers import normalize_telugu_numbers

# Independent Vowels
INDEPENDENT_VOWELS = {
    "అ": "a",
    "ఆ": "aa",
    "ఇ": "i",
    "ఈ": "ii",
    "ఉ": "u",
    "ఊ": "uu",
    "ఋ": "ru",
    "ౠ": "ruu",
    "ఌ": "lu",
    "ె": "e",
    "ఎ": "e",
    "ఏ": "ee",
    "ఐ": "ai",
    "ఒ": "o",
    "ఓ": "oo",
    "ఔ": "au",
}

# Dependent Vowel Signs (Matras)
DEPENDENT_VOWELS = {
    "\u0c3e": "aa",  # ా
    "\u0c3f": "i",   # ి
    "\u0c40": "ii",  # ీ
    "\u0c41": "u",   # ు
    "\u0c42": "uu",  # ూ
    "\u0c43": "ru",  # ృ
    "\u0c44": "ruu", # ౄ
    "\u0c46": "e",   # ె
    "\u0c47": "ee",  # ే
    "\u0c48": "ai",  # ై
    "\u0c4a": "o",   # ొ
    "\u0c4b": "oo",  # ో
    "\u0c4c": "au",  # ౌ
}

# Consonants
CONSONANTS = {
    "క": "k",
    "ఖ": "kh",
    "గ": "g",
    "ఘ": "gh",
    "ఙ": "ng",
    "చ": "c",
    "ఛ": "ch",
    "జ": "j",
    "ఝ": "jh",
    "ఞ": "ny",
    "ట": "tt",
    "ఠ": "tth",
    "డ": "dd",
    "ఢ": "ddh",
    "ణ": "nn",
    "త": "t",
    "థ": "th",
    "ద": "d",
    "ధ": "dh",
    "న": "n",
    "ప": "p",
    "ఫ": "ph",
    "బ": "b",
    "భ": "bh",
    "మ": "m",
    "య": "y",
    "ర": "r",
    "ఱ": "rr",
    "ల": "l",
    "ళ": "ll",
    "ఴ": "zh",
    "వ": "v",
    "శ": "sh",
    "ష": "ss",
    "స": "s",
    "హ": "h",
}

# Modifiers
MODIFIERS = {
    "\u0c01": "N",  # ఁ Candrabindu
    "\u0c02": "M",  # ం Anusvara (Sunna)
    "\u0c03": "H",  # ః Visarga
}

VIRAMA = "\u0c4d"  # ్ (Halant / Pollu)

# Complete list of valid Telugu phoneme symbols
valid_symbols = sorted(
    list(
        set(INDEPENDENT_VOWELS.values())
        | set(DEPENDENT_VOWELS.values())
        | set(CONSONANTS.values())
        | set(MODIFIERS.values())
    )
)


def telugu_word_to_phones(word: str):
    """Convert a single Telugu word into a list of phoneme symbols."""
    word = unicodedata.normalize("NFC", word.strip())
    phones = []
    chars = list(word)
    n = len(chars)
    i = 0

    while i < n:
        ch = chars[i]

        if ch in INDEPENDENT_VOWELS:
            phones.append(INDEPENDENT_VOWELS[ch])
            i += 1
        elif ch in CONSONANTS:
            c_phone = CONSONANTS[ch]
            phones.append(c_phone)
            # Look ahead for virama (halant) or dependent vowel matra
            if i + 1 < n and chars[i + 1] == VIRAMA:
                # Consonant cluster or word-final halant: suppress inherent 'a'
                i += 2
            elif i + 1 < n and chars[i + 1] in DEPENDENT_VOWELS:
                phones.append(DEPENDENT_VOWELS[chars[i + 1]])
                i += 2
            else:
                # Inherent vowel 'a'
                phones.append("a")
                i += 1
        elif ch in DEPENDENT_VOWELS:
            phones.append(DEPENDENT_VOWELS[ch])
            i += 1
        elif ch in MODIFIERS:
            phones.append(MODIFIERS[ch])
            i += 1
        elif ch == VIRAMA:
            i += 1
        else:
            # Skip zero-width joiners or stray punctuation
            i += 1

    return phones


def telugu_text_to_phones(text: str, include_word_spaces: bool = True):
    """
    Convert a full Telugu sentence into a list of phoneme symbols.
    Preserves word boundaries and punctuation pauses as 'sp' when include_word_spaces=True.
    """
    text = unicodedata.normalize("NFC", text)
    text = normalize_telugu_numbers(text)

    tokens = re.split(r"([,;.\-\?\!\s+])", text)
    phones = []

    for tok in tokens:
        if not tok:
            continue
        if re.match(r"^[,;.\-\?\!\s+]+$", tok):
            if include_word_spaces and phones and phones[-1] != "sp":
                phones.append("sp")
        else:
            w_phones = telugu_word_to_phones(tok)
            if w_phones:
                phones.extend(w_phones)

    # Trim leading/trailing 'sp'
    while phones and phones[0] == "sp":
        phones.pop(0)
    while phones and phones[-1] == "sp":
        phones.pop()

    return phones

