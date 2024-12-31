#!/usr/bin/env python

from pydantic import BaseModel

import torch
from fastapi import FastAPI, HTTPException
from transformers import MllamaForConditionalGeneration, BitsAndBytesConfig, AutoTokenizer

from gen import generate_responses, generate_mean_rates


MODEL = "./ludoviko"


class UserData(BaseModel):
    user: str
    user_prompt: str
    order: str


class Settings(BaseModel):
    system_prompt: str
    think_prompts: list

    rater_prompt: str
    rate_prompt: str

    temperature: float
    repetition_penalty: float
    top_p: float
    top_k: int

    max_new_tokens: int
    dynamic_token_shift: int
    thinking_tokens: int
    rate_tokens: int

    probe_num: int


class RequestBody(BaseModel):
    user_data: UserData
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
    user_data = request.user_data
    settings = request.settings

    responses = generate_responses(model, tokenizer, user_data, settings)
    rates = generate_mean_rates(model, tokenizer, responses, settings)

    for i, (response, rate) in enumerate(zip(responses, rates)):
        print(f"{i + 1} response ({rate:.2f}/10):\n{response}")

    best_idx = rates.index(max(rates))
    best_response = responses[best_idx]

    return {"response": best_response}
