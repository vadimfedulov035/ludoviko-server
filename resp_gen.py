#!/usr/bin/env python

import re

import numpy as np

from text_gen import TextGenerator
from proc import clean, check


class Responder(TextGenerator):

    def __init__(self, model, tokenizer, settings, user_data):
        # input as dialog -> auto token num deduction based on settings
        super().__init__(model, tokenizer, settings, user_data.dialog)
        self.user_data = user_data
        self._preprocess_dialog()


    def _preprocess_dialog(self):
        user_data = self.user_data

        user = user_data.user
        order = user_data.order
        dialog = user_data.dialog


    def _think(self, think_prompt, thought_chains):
        thoughts = []

        user_data = self.user_data
        settings = self.settings

        dialog = user_data.dialog
        probe_num = settings.probe_num

        user_prompt = think_prompt.format(dialog, *thought_chains)
        while len(thoughts) < probe_num:
            print(f"Generate try:", end=' ')

            thought_raw = self._generate(user_prompt)
            thought = clean(thought_raw, is_mutable=True)

            if check(thought):
                print("Success")
                thoughts.append(thought)
                print(thought)
            else:
                print("Failure")

        return thoughts


    def respond(self):
        settings = self.settings
        user_data = self.user_data

        dialog = user_data.dialog
        think_prompts = settings.think_prompts
        probe_num = settings.probe_num

        print("Step 1: Branching")
        thoughts = self._think(think_prompts[0], [dialog])
        thought_chains = np.array([thoughts])

        if len(think_prompts) == 1:
            return thoughts

        print("Step 2: Iterating")
        self.settings.probe_num = 1

        for think_prompt in think_prompts[1:]:
            thoughts.clear()

            for j in range(probe_num):
                thoughts += self._think(think_prompt, thought_chains[:, j])
            thought_chains = np.vstack((thought_chains, thoughts))

        responses = thoughts

        self.settings.probe_num = probe_num

        return responses
