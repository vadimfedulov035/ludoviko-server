#!/usr/bin/env python

import re

import torch


class TextGenerator():

    SYSTEM_TEMPLATE = "<|start_header_id|>system<|end_header_id|>{}<|eot_id|>"
    USER_TEMPLATE = "<|start_header_id|>user<|end_header_id|>{}<|eot_id|>"
    ASSISTANT_TEMPLATE = "<|start_header_id|>assistant<|end_header_id|>"

    def __init__(self, model, tokenizer, settings,
                 dialog: list[str], rate_mode: bool):
        """
        Initializes token num based on settings and rate mode.
        Sets -> dialog: list[str], dialog_str: str
        """
        self.model = model
        self.tokenizer = tokenizer
        self.settings = settings

        self.dialog = dialog
        for i, msg in enumerate(self.dialog):
            if "\n" in msg:
                self.dialog[i] = re.sub("\n+", " ", msg)
        self.dialog_str = "\n".join(dialog)

        # set token num based on the rate mode
        if rate_mode:
            self.settings.max_new_tokens = settings.rate_tokens
        else:
            self._set_token_num()


    def _set_token_num(self):
        """
        Sets static or dynamic token num,
        Dynamic token num is modified based by its shift value.
        """
        tokenizer = self.tokenizer
        settings = self.settings
        dialog = self.dialog

        # set static limit if specified
        if settings.max_new_tokens != 0:
            return

        # set dynamic limit if zero static limit
        encoded_inputs = tokenizer.encode(dialog[-1], return_tensors='pt')
        dynamic_token_num = encoded_inputs.size(1)
        max_new_tokens = dynamic_token_num + settings.dynamic_token_shift

        self.settings.max_new_tokens = max_new_tokens


    def _new_prompt(self, user_prompt: str) -> str:
        """
        Formats new prompt using
        settings system prompt and provided user prompt,
        based on class defined templates.
        """
        settings = self.settings

        prompt = ""
        prompt += self.SYSTEM_TEMPLATE.format(settings.system_prompt)
        prompt += self.USER_TEMPLATE.format(user_prompt)
        prompt += self.ASSISTANT_TEMPLATE

        if not "Rate: [number]" in prompt:
            print(f"Prompt:\n{prompt}")

        return prompt


    def _generate(self, user_prompt):
        """
        Calls new prompt formatting and generates response based on it.
        """
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
        Calculates Jaccard index to measure similarity
        between passed strings.
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
        using Jaccard index to check all phrases against each other.
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
        using Jaccard index to check all phrases against each other.
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


    def _is_sys_prompt(self, resp: str) -> tuple[bool, float]:
        """ Checks if response has discloses the system prompt
        using Jaccard index to response against the system prompt.
        """
        settings = self.settings

        system_prompt = settings.system_prompt

        is_sys_prompt = False
        jaccard_idx = self._calc_jaccard_idx(resp, system_prompt)
        if jaccard_idx > 0.1:
            is_sys_prompt = True

        return is_sys_prompt, jaccard_idx



    def _check(self, resp: str, msg: str) -> bool:
        """ Performs all the checks to ensure that response is valid """

        rate_str = r"([Rr]espondo(j)?(n)?\s(de\s.{1,24}\s)?(ne\s)?(\S+as))"
        is_rate = re.match(rate_str, resp) is not None
        is_web = "[RETEJO]" in resp
        is_short, resp_len, resp_min_len = self._is_short(resp, msg)
        is_repetitive, jaccard_idx_in = self._is_repetitive(resp)
        is_repetition, jaccard_idx_out = self._is_repetition(resp)
        is_sys_prompt, jaccard_idx_prompt = self._is_sys_prompt(resp)

        print(f"Is rate: {is_rate}")
        print(f"Is web: {is_web}")
        print(f"Is short: {is_short} ({resp_len:.2f}) < {resp_min_len:.2f})")
        print(f"Is repetitive: {is_repetitive} ({jaccard_idx_in:.2f})")
        print(f"Is repetition: {is_repetition} ({jaccard_idx_out:.2f})")
        print(f"Is system prompt: {is_sys_prompt} ({jaccard_idx_prompt:.2f})")

        is_malformed = is_rate or is_web or is_short
        is_tedious = is_repetitive or is_repetition or is_sys_prompt

        is_invalid = not (is_malformed or is_tedious)

        return is_invalid
