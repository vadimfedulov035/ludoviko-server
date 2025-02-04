"""
This module fixes common LLM mistypes as well as hallucinations.
- strips extra names (LLM naming itself, talking to itself further).
- strips partial sentences.
- adjusts punctuation.
"""


import re


def _strip_extra_names(text):
    text = re.sub(r'^\S{1,24}:\s', '', text)
    text = re.sub(r'\s?@\S{1,24}\s?', '', text)
    text = re.sub(r'(\w|_|-){1,24}:.*', '', text)

    return text


def _strip_partial_sentences(text):
    if text:
        text = text[0].upper() + text[1:]

    last_index = max(
        (text.rfind(char) for char in '.!?'),
        default=-1
    )
    if last_index != -1 and last_index > 10:
        text = text[:last_index + 1]

    return text


def _adjust_punctuation(text):
    text = re.sub(r' \.', '.', text)

    text = re.sub(r'\s*,\s*', ', ', text)
    pattern = r'\s*([.!?]\s*)(\S)'

    def fix(m):
        punct = m.group(1)
        letter = m.group(2)
        return f'{punct.strip()} {letter.upper()}'

    text = re.sub(pattern, fix, text)

    return text


def clean(text):
    """ Cleans LLM generated text and fixes some hallucinations. """
    text = text.split("EOS")[0]

    text = _strip_extra_names(text)
    text = _strip_partial_sentences(text)
    text = _adjust_punctuation(text)

    return text
