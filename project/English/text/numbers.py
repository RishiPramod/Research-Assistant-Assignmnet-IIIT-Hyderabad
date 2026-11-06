"""Number normalization for English TTS (uses inflect if available, with built-in fallback)."""

import re

try:
    import inflect
    _inflect = inflect.engine()
except ImportError:
    _inflect = None

_comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")
_decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
_pounds_re = re.compile(r"£([0-9\,]*[0-9]+)")
_dollars_re = re.compile(r"\$([0-9\.\,]*[0-9]+)")
_ordinal_re = re.compile(r"[0-9]+(st|nd|rd|th)")
_number_re = re.compile(r"[0-9]+")

_EN_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
]
_EN_TENS = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
]
_EN_ORDINAL_EXACT = {
    1: "first", 2: "second", 3: "third", 5: "fifth", 8: "eighth", 9: "ninth", 12: "twelfth",
}


def _int_to_en_words(n: int) -> str:
    if n < 0:
        return "minus " + _int_to_en_words(-n)
    if n < 20:
        return _EN_ONES[n]
    if n < 100:
        t, r = divmod(n, 10)
        return _EN_TENS[t] if r == 0 else f"{_EN_TENS[t]} {_EN_ONES[r]}"
    if n < 1000:
        h, r = divmod(n, 100)
        return f"{_EN_ONES[h]} hundred" if r == 0 else f"{_EN_ONES[h]} hundred {_int_to_en_words(r)}"
    if n < 1000000:
        th, r = divmod(n, 1000)
        return f"{_int_to_en_words(th)} thousand" if r == 0 else f"{_int_to_en_words(th)} thousand {_int_to_en_words(r)}"
    return " ".join(_EN_ONES[int(d)] for d in str(n))


def _remove_commas(m):
    return m.group(1).replace(",", "")


def _expand_decimal_point(m):
    return m.group(1).replace(".", " point ")


def _expand_dollars(m):
    match = m.group(1)
    parts = match.split(".")
    if len(parts) > 2:
        return match + " dollars"
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s, %s %s" % (dollars, dollar_unit, cents, cent_unit)
    elif dollars:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        return "%s %s" % (dollars, dollar_unit)
    elif cents:
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s" % (cents, cent_unit)
    else:
        return "zero dollars"


def _expand_ordinal(m):
    if _inflect is not None:
        return _inflect.number_to_words(m.group(0))
    val = int(re.sub(r"(st|nd|rd|th)$", "", m.group(0)))
    if val in _EN_ORDINAL_EXACT:
        return _EN_ORDINAL_EXACT[val]
    w = _int_to_en_words(val)
    if w.endswith("y"):
        return w[:-1] + "ieth"
    return w + "th"


def _expand_number(m):
    num = int(m.group(0))
    if _inflect is not None:
        if 1000 < num < 3000:
            if num == 2000:
                return "two thousand"
            elif 2000 < num < 2010:
                return "two thousand " + _inflect.number_to_words(num % 100)
            elif num % 100 == 0:
                return _inflect.number_to_words(num // 100) + " hundred"
            else:
                return _inflect.number_to_words(
                    num, andword="", zero="oh", group=2
                ).replace(", ", " ")
        else:
            return _inflect.number_to_words(num, andword="").replace("-", " ").replace(", ", " ")
    else:
        if 1000 < num < 3000:
            if num == 2000:
                return "two thousand"
            elif 2000 < num < 2010:
                return "two thousand " + _int_to_en_words(num % 100)
            elif num % 100 == 0:
                return _int_to_en_words(num // 100) + " hundred"
            else:
                hi, lo = divmod(num, 100)
                lo_str = f"oh {_int_to_en_words(lo)}" if lo < 10 else _int_to_en_words(lo)
                return f"{_int_to_en_words(hi)} {lo_str}"
        return _int_to_en_words(num)


def normalize_numbers(text):
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_pounds_re, r"\1 pounds", text)
    text = re.sub(_dollars_re, _expand_dollars, text)
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    text = re.sub(_number_re, _expand_number, text)
    return text
