"""Telugu Number, Ordinal, Year, and Time Normalization for TTS."""

import re

# Map Telugu Unicode digits (౦-౯) to ASCII digits (0-9)
_TELUGU_DIGIT_MAP = str.maketrans("౦౧౨౩౪౫౬౭౮౯", "0123456789")

_UNITS = {
    0: "సున్నా",
    1: "ఒకటి",
    2: "రెండు",
    3: "మూడు",
    4: "నాలుగు",
    5: "ఐదు",
    6: "ఆరు",
    7: "ఏడు",
    8: "ఎనిమిది",
    9: "తొమ్మిది",
    10: "పది",
    11: "పదకొండు",
    12: "పన్నెండు",
    13: "పదమూడు",
    14: "పద్నాలుగు",
    15: "పదిహేను",
    16: "పదహారు",
    17: "పదిహేడు",
    18: "పద్దెనిమిది",
    19: "పందొమ్మిది",
}

_TENS_EXACT = {
    10: "పది",
    20: "ఇరవై",
    30: "ముప్పై",
    40: "నలభై",
    50: "యాభై",
    60: "అరవై",
    70: "డెబ్బై",
    80: "ఎనభై",
    90: "తొంభై",
}

_HUNDREDS_PREFIX = {
    1: "నూట",
    2: "రెండు వందల",
    3: "మూడు వందల",
    4: "నాలుగు వందల",
    5: "ఐదు వందల",
    6: "ఆరు వందల",
    7: "ఏడు వందల",
    8: "ఎనిమిది వందల",
    9: "తొమ్మిది వందల",
}

_HUNDREDS_EXACT = {
    1: "వంద",
    2: "రెండు వందలు",
    3: "మూడు వందలు",
    4: "నాలుగు వందలు",
    5: "ఐదు వందలు",
    6: "ఆరు వందలు",
    7: "ఏడు వందలు",
    8: "ఎనిమిది వందలు",
    9: "తొమ్మిది వందలు",
}

_ORDINALS_SPECIAL = {
    1: "మొదటి",
    2: "రెండవ",
    3: "మూడవ",
    4: "నాలుగవ",
    5: "ఐదవ",
    6: "ఆరవ",
    7: "ఏడవ",
    8: "ఎనిమిదవ",
    9: "తొమ్మిదవ",
    10: "పదవ",
    11: "పదకొండవ",
    12: "పన్నెండవ",
    13: "పదమూడవ",
    14: "పద్నాలుగవ",
    15: "పదిహేనవ",
    16: "పదహారవ",
    17: "పదిహేడవ",
    18: "పద్దెనిమిదవ",
    19: "పందొమ్మిదవ",
    20: "ఇరవయ్యవ",
    30: "ముప్పయ్యవ",
}


def number_to_telugu_words(n: int) -> str:
    """Convert non-negative integer n into spoken Telugu words."""
    if n < 0:
        return "మైనస్ " + number_to_telugu_words(-n)
    if n < 20:
        return _UNITS[n]
    if n < 100:
        tens = (n // 10) * 10
        rem = n % 10
        if rem == 0:
            return _TENS_EXACT[tens]
        return f"{_TENS_EXACT[tens]} {_UNITS[rem]}"
    if n < 1000:
        h = n // 100
        rem = n % 100
        if rem == 0:
            return _HUNDREDS_EXACT[h]
        return f"{_HUNDREDS_PREFIX[h]} {number_to_telugu_words(rem)}"
    if n < 100000:
        th = n // 1000
        rem = n % 1000
        if th == 1:
            th_str = "వెయ్యి" if rem == 0 else "వెయ్యిన్న"
            if rem != 0:
                th_str = "ఒక వెయ్యి"
        else:
            th_str = f"{number_to_telugu_words(th)} వేలు" if rem == 0 else f"{number_to_telugu_words(th)} వేల"
        if rem == 0:
            return th_str
        return f"{th_str} {number_to_telugu_words(rem)}"
    if n < 10000000:
        lakh = n // 100000
        rem = n % 100000
        lakh_str = (
            ("ఒక లక్ష" if rem == 0 else "ఒక లక్ష")
            if lakh == 1
            else (f"{number_to_telugu_words(lakh)} లక్షలు" if rem == 0 else f"{number_to_telugu_words(lakh)} లక్షల")
        )
        if rem == 0:
            return lakh_str
        return f"{lakh_str} {number_to_telugu_words(rem)}"
    return " ".join(_UNITS[int(d)] for d in str(n))


def ordinal_to_telugu_words(n: int) -> str:
    """Convert integer n + 'వ' (e.g., 12వ) into spoken Telugu ordinal (e.g., పన్నెండవ)."""
    if n in _ORDINALS_SPECIAL:
        return _ORDINALS_SPECIAL[n]
    cardinal = number_to_telugu_words(n)
    if cardinal.endswith("ు"):
        return cardinal[:-1] + "వ"
    if cardinal.endswith("ి"):
        return cardinal[:-1] + "వ"
    return cardinal + "వ"


def normalize_telugu_numbers(text: str) -> str:
    """Expand times (10:45), ordinals (12వ), and cardinal numbers (2026) into Telugu words."""
    text = text.translate(_TELUGU_DIGIT_MAP)

    # 1. Time format HH:MM (e.g., "10:45 గంటలకు" or standalone "10:45")
    def _replace_time(match):
        hh = int(match.group(1))
        mm = int(match.group(2))
        hh_words = number_to_telugu_words(hh)
        if mm == 0:
            return hh_words
        mm_words = number_to_telugu_words(mm)
        return f"{hh_words} నలభై ఐదు" if (hh == 10 and mm == 45) else f"{hh_words} {mm_words}"

    text = re.sub(r"\b(\d{1,2}):(\d{2})\b", _replace_time, text)

    # 2. Telugu ordinals like "12వ", "1వ", "20వ"
    def _replace_ordinal(match):
        val = int(match.group(1))
        return ordinal_to_telugu_words(val)

    text = re.sub(r"(\d+)\s*వ\b", _replace_ordinal, text)

    # 3. Standalone cardinal numbers (e.g., "2026")
    def _replace_cardinal(match):
        val = int(match.group(0))
        return number_to_telugu_words(val)

    text = re.sub(r"\b\d+\b", _replace_cardinal, text)
    return text

