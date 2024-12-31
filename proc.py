import re


def convert_to_int(s):
    s = s.strip()

    if '/' in s:
        s = s.split('/')[0]
    if 'el' in s:
        s = s.split('el')[0]

    digits = ''.join(char for char in s if char.isdigit())
    if not digits:
        print("No digits")
        return -1

    result = int(digits)

    if result < 0:
        print(f"Too low result: {result}")
        result = -1
    elif result > 10:
        print(f"Too high result: {result}")
        result = -1

    return result


def _fix_extra_phrases(text):
    text = re.sub(r'^@(.){1,24}\s', '', text)
    text = re.sub(r'^(.){1,24}:\s', '', text)
    text = re.sub(r'(\w|_|-){1,24}:.*', '', text)

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


def _fix_partiality(text):
    if text:
        text = text[0].upper() + text[1:]

    last_index = max(
        (text.rfind(char) for char in '.!?'),
        default=-1
    )
    if last_index != -1 and last_index > 10:
        text = text[:last_index + 1]

    return text


def _fix_spacing(text):
    text = re.sub(r'\s*,\s*', ', ', text)

    pattern = r'\s*([.!?]\s*)(\S)'
    def fix(m):
        punct = m.group(1)
        letter = m.group(2)
        return f'{punct.strip()} {letter.upper()}'
    text = re.sub(pattern, fix, text)

    return text


def clean(text):
    text = _fix_extra_phrases(text)

    text = _fix_spacing(text)
    text = _fix_partiality(text)

    #if not tl_mode:
    text = _fix_extra_questions(text)

    return text


def check(text, num):
    is_short = len(text) < num / 3
    is_web = "[RETEJO]" in text

    is_malformed = is_short or is_web

    return not is_malformed
