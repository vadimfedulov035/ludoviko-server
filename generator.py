"""
This module defines TextGenerator superclass for Responder and Rater classes.
It sets token number based on passed settings.
It defines self.generate() and self.check() methods for subclass use.
"""


from typing import Optional

import re

import torch



class Generator():

    """
    Superclass for Responder and Rater classes.
    self.generate(user_prompt: str) generates LLM response.
    self.check(resp: str, msg: str) checks message (msg is for length check).
    """

    SYSTEM_TEMPLATE = "<|start_header_id|>system<|end_header_id|>{}<|eot_id|>"
    USER_TEMPLATE = "<|start_header_id|>user<|end_header_id|>{}<|eot_id|>"
    ASSISTANT_TEMPLATE = "<|start_header_id|>assistant<|end_header_id|>"

    def __init__(self, llm, settings, dialog: list[str], responses: Optional[list[str]]):
        """
        Initializes token number based on settings.
        Sets -> self.dialog: list[str], self.dialog_str: str
        """
        self.model, self.tokenizer = llm.values()
        self.settings = settings

        self.dialog = dialog
        for i, msg in enumerate(self.dialog):
            if "\n" in msg:
                self.dialog[i] = re.sub("\n+", " ", msg)
        self.dialog_str = "\n".join(dialog)

        self.responses = responses

        self._set_token_num()


    def _set_token_num(self):
        """
        Sets settings.max_new_tokens to static/dynamic/rate token number.
        Static token number <- settings.max_new_tokens.
        Dynamic token number <- length + settings.dynamic_token_shift.
        Rate token number <- settings.rate_tokens if all is set to 0.
        """
        tokenizer = self.tokenizer
        settings = self.settings
        dialog = self.dialog

        max_new_tokens = 0
        static = settings.max_new_tokens != 0
        dynamic = settings.dynamic_token_shift != 0
        if static:
            max_new_tokens = settings.max_new_tokens
        elif not static and dynamic:
            encoded_inputs = tokenizer.encode(dialog[-1], return_tensors='pt')
            dynamic_token_num = encoded_inputs.size(1)
            max_new_tokens = dynamic_token_num + settings.dynamic_token_shift
        else:
            max_new_tokens = settings.rate_tokens

        self.settings.max_new_tokens = max_new_tokens


    def _new_prompt(self, user_prompt: str) -> str:
        """
        Formats new prompt using
        settings' system prompt and provided user prompt,
        based on class defined templates.
        """
        settings = self.settings

        prompt = ""
        prompt += self.SYSTEM_TEMPLATE.format(settings.system_prompt)
        prompt += self.USER_TEMPLATE.format(user_prompt)
        prompt += self.ASSISTANT_TEMPLATE

        if "Rate: [number]" not in prompt:
            print(f"Prompt:\n{prompt}")

        return prompt


    def generate(self, user_prompt):
        """ Generates response based on user prompt. """
        model = self.model
        tokenizer = self.tokenizer
        settings = self.settings

        prompt = self._new_prompt(user_prompt)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                    **inputs,
                    do_sample          = True,
                    bos_token_id       = tokenizer.bos_token_id,
                    eos_token_id       = tokenizer.eos_token_id,
                    pad_token_id       = tokenizer.pad_token_id,
                    temperature        = settings.temperature,
                    repetition_penalty = settings.repetition_penalty,
                    top_p              = settings.top_p,
                    top_k              = settings.top_k,
                    max_new_tokens     = settings.max_new_tokens
            )
            response_raw = tokenizer.decode(outputs[0],
                                            skip_special_tokens=True)
            response_raw = response_raw.split("assistant")[-1].strip()

            return response_raw


    def _is_short(self, resp: str, msg: str) -> tuple[bool, float, float]:
        """ Checks if response is less than 0.75 of message length. """
        resp_len, msg_len = len(resp), len(msg)
        resp_min_len = msg_len * 0.75
        return resp_len < resp_min_len, resp_len, resp_min_len


    def _split_to_phrases(self, text: str) -> list[str]:
        """ Splits phrases based on delimeters. """
        delimiters = (".", ",", "!", "?", ";")
        pattern = '|'.join(map(re.escape, delimiters))
        return re.split(pattern, text)


    def _calc_jaccard_idx(self, str1: str, str2: str) -> float:
        """
        Calculates Jaccard index to measure similarity between passed strings.
        """
        jaccard_idx = 0

        set1 = set(str1.split())
        set2 = set(str2.split())

        intersection_length = len(set1.intersection(set2))
        union_length = len(set1.union(set2))

        if union_length != 0:
            jaccard_idx = intersection_length / union_length

        return jaccard_idx


    def _is_repetitive(self, resp: str) -> tuple[bool, float]:
        """
        Checks if response has repetitions within itself
        using Jaccard index to check all internal phrases against each other.
        """
        jaccard_idx = 0
        phrases = self._split_to_phrases(resp)

        for i in range(len(phrases)):
            for j in range(i + 1, len(phrases)):
                jaccard_idx = self._calc_jaccard_idx(phrases[i], phrases[j])
                if jaccard_idx > 0.5:
                    return True, jaccard_idx

        return False, jaccard_idx


    def _is_repetition(self, resp: str) -> tuple[bool, float]:
        """
        Checks if response has repetitions with other messsages in the dialog
        using Jaccard index to check message phrases against external phrases.
        """
        dialog = self.dialog

        jaccard_idx = 0
        resp_phrases = self._split_to_phrases(resp)

        for msg in dialog:
            msg_phrases = self._split_to_phrases(msg)

            for msg_phrase in msg_phrases:
                for resp_phrase in resp_phrases:
                    jaccard_idx = self._calc_jaccard_idx(msg_phrase,
                                                         resp_phrase)
                    if jaccard_idx > 0.5:
                        return True, jaccard_idx

        return False, jaccard_idx


    def check(self, resp: str, msg: str) -> tuple[bool, str]:
        """
        Checks if LLM response is web-related, short, repetitive
        or is repetition of some message in dialog.
        """
        msg = ""

        is_web = "[RETEJO]" in resp
        is_short, resp_len, resp_min_len = self._is_short(resp, msg)
        is_repetitive, jaccard_idx_in = self._is_repetitive(resp)
        is_repetition, jaccard_idx_out = self._is_repetition(resp)

        if is_web:
            msg += "Web: [RETEJO] in response\n"
        if is_short:
            msg += f"Short: ({resp_len:.2f}) < {resp_min_len:.2f})\n"
        if is_repetitive:
            msg += f"Repetitive: ({jaccard_idx_in:.2f}) > 0.5\n"
        if is_repetition:
            msg += f"Repetition: ({jaccard_idx_out:.2f}) > 0.5\n"

        is_malformed = is_web or is_short
        is_tedious = is_repetitive or is_repetition
        is_valid = not (is_malformed or is_tedious)

        return is_valid, msg
