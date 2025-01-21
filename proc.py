import re


def _fix_extra_names(text):
    text = re.sub(r'^\S{1,24}:\s', '', text)
    text = re.sub(r'@\S{1,24}\s', '', text)
    text = re.sub(r'(\w|_|-){1,24}:.*', '', text)

    return text


def _fix_sentences(text):
    if text:
        text = text[0].upper() + text[1:]

    last_index = max(
        (text.rfind(char) for char in '.!?'),
        default=-1
    )
    if last_index != -1 and last_index > 10:
        text = text[:last_index + 1]

    return text


def _fix_punctuation(text):
    text = re.sub(r'\.\. \.', '...', text)
    text = re.sub(r'\. \.\.', '...', text)

    text = re.sub(r'\s*,\s*', ', ', text)
    pattern = r'\s*([.!?]\s*)(\S)'
    def fix(m):
        punct = m.group(1)
        letter = m.group(2)
        return f'{punct.strip()} {letter.upper()}'
    text = re.sub(pattern, fix, text)

    return text


def _fix_extra_questions(text):
    result = ""

    q_num = 0
    sentence, delimeter = "", ""
    for i, x in enumerate(re.split(r'([.!?])', text)):
        if i % 2 == 0:
            sentence = x
            continue
        if (delimeter := x) == '?':
            q_num += 1
            if q_num > 2:
                continue
        result += sentence + delimeter

    return result


def is_mostly_cyrillic(text):
    total_letters = 0
    cyrillic_letters = 0

    for char in text:
        if char.isalpha():
            total_letters += 1
            if 'а' <= char <= 'я' or 'А' <= char <= 'Я':
                cyrillic_letters += 1

    if total_letters == 0:
        return True

    return cyrillic_letters > total_letters / 2


def clean(text, is_mutable):
    text = text.split("EOS")[0]

    text = _fix_extra_names(text)
    text = _fix_sentences(text)
    text = _fix_punctuation(text)

    if is_mutable:
        text = _fix_extra_questions(text)

    return text


def check(text):
    is_short = len(text) < 25
    is_web = "[RETEJO]" in text
    is_too_cyrillic = is_mostly_cyrillic(text)

    is_malformed = is_short or is_web or is_too_cyrillic

    return not is_malformed
