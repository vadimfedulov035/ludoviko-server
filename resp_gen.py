#!/usr/bin/env python

import numpy as np
from numpy.typing import NDArray

from text_gen import TextGenerator
from proc import clean


class Responder(TextGenerator):

    def __init__(self, model, tokenizer, settings, dialog: list[str]):
        """
        Superclass initializes token num based on settings and rate mode.
        Sets dialog: list[str], dialog_str: str
        """
        super().__init__(model, tokenizer, settings, dialog, rate_mode=False)


    def _think(self, think_prompt: str, thought_chain: NDArray) -> list[str]:
        """
        Generates thought based on provided think prompt
        formatted with expanded thought chain.
        Note: Every think prompt should have '{}'-num equal to its index.
        """
        thoughts = []

        dialog = self.dialog
        dialog_str = self.dialog_str
        settings = self.settings

        batch_size = settings.batch_size

        user_prompt = think_prompt.format(dialog_str, *thought_chain)
        while len(thoughts) < batch_size:
            print(f"Generate try:", end=' ')

            thought_raw = self._generate(user_prompt)
            thought = clean(thought_raw)

            if self._check(thought, dialog[-1]):
                print("Success")
                thoughts.append(thought)
                print(thought)
            else:
                print("Failure")

        return thoughts


    def respond(self) -> list[str]:
        """
        Implements Chain of Thought algorithm.
        Creates and independently continues parrallel thought chains.
        Treats thoughts as responses if think prompts finished.
        """

        settings = self.settings
        dialog_str = self.dialog_str

        think_prompts = settings.think_prompts
        batch_size = settings.batch_size

        # for the initial think prompt
        # batch generate first thoughts in the chains
        print("Step 1: CoT start")
        thoughts = self._think(think_prompts[0], np.array([dialog_str]))
        if len(think_prompts) == 1:
            return thoughts
        thought_chains = np.array([thoughts])

        # for every non-initial think prompt
        # accumulate next thoughts to continue the chains
        print("Step 2: CoT continue")
        self.settings.batch_size = 1

        for think_prompt in think_prompts[1:]:
            thoughts.clear()

            for j in range(batch_size):
                thoughts += self._think(think_prompt, thought_chains[:, j])
            thought_chains = np.vstack((thought_chains, thoughts))

        responses = thoughts

        # revert base batch size for object reuse (not needed)
        self.settings.batch_size = batch_size

        return responses
