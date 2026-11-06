"""
Defines the set of symbols used in text input to the Telugu FastSpeech 2 model.
Includes Telugu phoneme symbols (@a, @aa, @k, @tt, etc.), ARPAbet, and silence markers (@sp, @spn, @sil).
"""

from text import cmudict, pinyin, telugu_phonemes

_pad = "_"
_punctuation = "!'(),.:;? "
_special = "-"
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_silences = ["@sp", "@spn", "@sil"]

# Prepend "@" to phoneme symbols to ensure uniqueness:
_telugu = ["@" + s for s in telugu_phonemes.valid_symbols]
_arpabet = ["@" + s for s in cmudict.valid_symbols]
_pinyin = ["@" + s for s in pinyin.valid_symbols]

# Export all symbols:
symbols = (
    [_pad]
    + list(_special)
    + list(_punctuation)
    + list(_letters)
    + _telugu
    + _arpabet
    + _pinyin
    + _silences
)
