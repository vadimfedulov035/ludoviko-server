import re

from text_gen import TextGenerator


class Rater(TextGenerator):

    def __init__(self, model, tokenizer, settings,
                 dialog: list[str], responses: list[str]):
        """
        Superclass initializes token num based on settings and rate mode.
        Sets -> dialog: list[str], dialog_str: str.
        """
        super().__init__(model, tokenizer, settings, dialog, rate_mode=True)
        self.responses = responses


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


    def _rate_avg(self, response):
        """ Calculates average rate from <rate_num> of rates. """
        rates = []

        settings = self.settings
        dialog_str = self.dialog_str

        rate_prompt = settings.rate_prompt
        rate_num = settings.rate_num

        user_prompt = rate_prompt.format(dialog_str, response)
        while len(rates) < rate_num:
            print(f"Rate try:", end=' ')
            resp = self._generate(user_prompt)
            rate = self._to_rate(resp)
            if rate != -1:
                rates.append(rate)
                print(f"Success ({rate}/10)")
            else:
                print(f"Failure {resp}")
        mean_rate = sum(rates) / len(rates)
        return mean_rate


    def rate(self):
        """ Calculates average rates for all responses"""
        rates = []

        responses = self.responses
        for response in responses:
            rate = self._rate_avg(response)
            rates.append(rate)

        return rates
