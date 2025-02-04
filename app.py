#!/usr/bin/env python

import torch
from fastapi import FastAPI
from transformers import MllamaForConditionalGeneration, BitsAndBytesConfig, AutoTokenizer
from pydantic import BaseModel

from resp_gen import Responder
from rate_gen import Rater


MODEL = "./ludoviko"


class Settings(BaseModel):
    system_prompt: str
    think_prompts: list
    rate_prompt: str

    temperature: float
    repetition_penalty: float
    top_p: float
    top_k: int

    max_new_tokens: int
    dynamic_token_shift: int
    rate_tokens: int

    batch_size: int
    rate_num: int


class RequestBody(BaseModel):
    dialog: list[str]
    settings: Settings


tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = MllamaForConditionalGeneration.from_pretrained(
    MODEL,
    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    torch_dtype=torch.float16,
    device_map="cuda"
)


app = FastAPI()


@app.post("/api/chat")
async def chat(request: RequestBody):
    dialog = request.dialog
    settings = request.settings

    responder = Responder(model, tokenizer, settings, dialog)
    responses = responder.respond()

    rater = Rater(model, tokenizer, settings, dialog, responses)
    rates = rater.rate()

    best_idx = rates.index(max(rates))
    best_response = responses[best_idx]

    for rate, response in zip(rates, responses):
        print(f"Rate: {rate:.1f}/10\n{response}\n")

    return {"response": best_response}
