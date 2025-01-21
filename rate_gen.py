import re

from text_gen import TextGenerator
from proc import clean, check


class Rater(TextGenerator):

    def __init__(self, model, tokenizer, settings, user_data, responses):
        # input as None -> initialize max_token_num with rate_tokens
        super().__init__(model, tokenizer, settings, None)
        self.user_data = user_data
        self.responses = responses

    def _to_rate(self, text):
        text = re.sub(r"^[Rr]ate:\s?", "", text)
        text = re.split(r"(?:\s|\.|/|el)", text)[0]

        digits = ''.join(char for char in text if char.isdigit())
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


    def _rate_mean(self, dialog, response):
        rates = []

        settings = self.settings

        rate_prompt = settings.rate_prompt
        probe_num = settings.probe_num

        user_prompt = rate_prompt.format(dialog, response)
        while len(rates) < probe_num:
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
        rates = []
        responses = self.responses

        user_data = self.user_data
        settings = self.settings

        dialog = user_data.dialog
        probe_num = settings.probe_num

        for response in responses:
            while len(rates) < probe_num:
                rate = self._rate_mean(dialog, response)
                rates.append(rate)

        return rates
