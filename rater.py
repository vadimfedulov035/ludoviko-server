"""
This module defines Rater class on top of Generator superclass.
It relies on superclass initialization.
It defines self.rate() for external usage.
"""


import re

from generator import Generator


class Rater(Generator):

    """
    Rater uses Generator initialize method.
    Sets -> self.dialog: list[str], self.dialog_str: str
    self.rate() makes LLM rate responses as last part of dialog.
    """

    def _to_rate(self, resp: str):
        """ Converts response to rate. """
        resp = re.sub(r"^[Rr]ate:\s?", "", resp)
        resp = re.split(r"(?:\s|\.|/|el)", resp)[0]

        digits = ''.join(char for char in resp if char.isdigit())
        if not digits:
            print("No digits")
            return -1

        rate = int(digits)
        if rate < 0:
            print(f"Too low rate: {rate}")
            rate = -1
        elif rate > 10:
            print(f"Too high rate: {rate}")
            rate = -1

        return rate


    def _rate_avg(self, resp: str):
        """ Calculates average rate from <rate_num> of rates. """
        rates = []

        settings = self.settings
        dialog_str = self.dialog_str

        rate_prompt = settings.rate_prompt
        rate_num = settings.rate_num

        user_prompt = rate_prompt.format(dialog_str, resp)
        while len(rates) < rate_num:
            resp = self.generate(user_prompt)
            rate = self._to_rate(resp)
            if rate != -1:
                rates.append(rate)
            else:
                print(f"Rate failure: {resp}")
        mean_rate = sum(rates) / len(rates)
        return mean_rate


    def rate(self):
        """ Calculates average rates for all responses. """
        rates = []

        responses = self.responses
        if responses is None:
            return rates

        for response in responses:
            rate = self._rate_avg(response)
            rates.append(rate)

        return rates
