#!/usr/bin/env python

import re

import torch

from proc import clean, check


class TextGenerator():

    SYSTEM_TEMPLATE = "<|start_header_id|>system<|end_header_id|>{}<|eot_id|>"
    USER_TEMPLATE = "<|start_header_id|>user<|end_header_id|>{}<|eot_id|>"
    ASSISTANT_TEMPLATE = "<|start_header_id|>assistant<|end_header_id|>"

    def __init__(self, model, tokenizer, settings, data):
        self.model = model
        self.tokenizer = tokenizer
        self.settings = settings

        self._set_token_num(data)


    def _set_token_num(self, data):
        tokenizer = self.tokenizer
        settings = self.settings

        # data passed as None means rate mode
        if data is None:
            self.settings.max_new_tokens = settings.rate_tokens
            return

        if settings.max_new_tokens != 0:
            return

        msg = re.split(r'(?:^|\n)(?:\S{1,24}:\s)', data)[-1]
        encoded_inputs = tokenizer.encode(msg, return_tensors='pt')
        dynamic_token_num = encoded_inputs.size(1)

        max_new_tokens = dynamic_token_num + settings.dynamic_token_shift

        self.settings.max_new_tokens = max_new_tokens


    def _new_prompt(self, user_prompt):
        settings = self.settings

        prompt = ""
        prompt += self.SYSTEM_TEMPLATE.format(settings.system_prompt)
        prompt += self.USER_TEMPLATE.format(user_prompt)
        prompt += self.ASSISTANT_TEMPLATE

        #if not "Rate: [number]" in prompt:
        #    print(f"Prompt:\n{prompt}")

        return prompt


    def _generate(self, user_prompt):
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
