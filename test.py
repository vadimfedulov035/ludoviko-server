import re


def _fix_tags(text):
    thinking_tag = "Thinking: "
    response_tag = "Response: "

    i = text.find(thinking_tag)
    if i == -1:
        return ""
    i = text.find(response_tag)
    if i == -1:
        return ""

    i += len(response_tag)
    text = text[i:]

    text = re.sub(rf"{thinking_tag}.*", '', text)
    text = re.sub(rf"{response_tag}.*", '', text)

    return text


text = "Thinking: Hm... Response: Yes Thinking: Ohhh... Response: No"
text = _fix_tags(text)
print(text)
