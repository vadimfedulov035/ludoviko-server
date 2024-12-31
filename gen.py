#!/usr/bin/env python

import re

import torch
from fastapi import FastAPI, HTTPException
from transformers import MllamaForConditionalGeneration, BitsAndBytesConfig, AutoTokenizer

from pydantic import BaseModel

from proc import clean, check, convert_to_int


RATE_TOKENS = 25

SYSTEM_TEMPLATE = "<|start_header_id|>system<|end_header_id|>{}<|eot_id|>"
USER_TEMPLATE = "<|start_header_id|>user<|end_header_id|>{}<|eot_id|>"
ASSISTANT_TEMPLATE = "<|start_header_id|>assistant<|end_header_id|>"


def set_dynamic_token_num(tokenizer, prompt):
    encoded_inputs = tokenizer.encode(prompt, return_tensors='pt')
    dynamic_token_num = encoded_inputs.size(1)

    return dynamic_token_num


def new_prompt(system_prompt, user_prompt):
    prompt = ""
    prompt += SYSTEM_TEMPLATE.format(system_prompt)
    prompt += USER_TEMPLATE.format(user_prompt)
    prompt += ASSISTANT_TEMPLATE

    return prompt


def generate(model, tokenizer, prompt, settings):
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


def _generate_thoughts(model, tokenizer, prompt, settings, i):
    thoughts = []

    while len(thoughts) < settings.probe_num:
        print(f"Generate try {i + 1}:", end=' ')

        thought_raw = generate(model, tokenizer, prompt, settings)
        thought_clean = clean(thought_raw)

        if check(thought_clean, settings.max_new_tokens):
            thoughts.append(thought_clean)
            print("Success")
            print(thought_clean)
        else:
            print("Failure")
        i += 1

    return i, thoughts


def generate_responses(model, tokenizer, user_data, settings):

    system_prompt = settings.system_prompt
    think_prompts = settings.think_prompts

    user = user_data.user
    user_prompt = user_data.user_prompt
    order = user_data.order

    if order != "":
        user_prompt = re.sub(rf"{user}: {order}", "", user_prompt)

    if settings.max_new_tokens == 0:
        dynamic_token_num = set_dynamic_token_num(tokenizer, user_prompt)
        settings.max_new_tokens = dynamic_token_num + settings.dynamic_token_shift
    max_new_tokens = settings.max_new_tokens

    print("Step 1: Branching")
    i = 0
    settings.max_new_tokens = settings.thinking_tokens
    prompt = new_prompt(system_prompt, user_prompt + think_prompts[0])
    i, thoughts = _generate_thoughts(model, tokenizer, prompt, settings, i)

    if len(think_prompts) < 2:
        responses = thoughts
        return responses

    print("Step 2: Iterating")
    probe_num = settings.probe_num
    settings.probe_num = 1
    thoughts_new = []
    for i, think_prompt in enumerate(think_prompts[1:]):
        last_iter = i == (len(think_prompts[:1]) - 1)
        if last_iter:
            settings.max_new_tokens = max_new_tokens

        thoughts_new = []
        for thought in thoughts:
            prompt = new_prompt(system_prompt,
                                user_prompt + thought + think_prompt)
            i, thought_new = _generate_thoughts(model, tokenizer,
                                             prompt, settings, i)
            thoughts_new.extend(thought_new)
        
        thoughts_tmp = []
        for thought, thought_new in zip(thoughts, thoughts_new):
            thoughts_tmp.append(thought + thought_new)
        thoughts = thoughts_tmp

    settings.probe_num = probe_num

    thought_chain = thoughts
    responses = thoughts_new

    return responses


def _generate_mean_rate(model, tokenizer, prompt, settings, i):
    rates = []
    while len(rates) < settings.probe_num:
        print(f"Rate try {i + 1}:", end=' ')
        resp = generate(model, tokenizer, prompt, settings)
        rate = convert_to_int(resp)
        if rate != -1:
            rates.append(rate)
            print(f"Success ({rate}/10)")
        else:
            print(f"Failure {resp}")
        i += 1
    mean_rate = sum(rates) / len(rates)
    return i, mean_rate


def generate_mean_rates(model, tokenizer, responses, settings):
    rates = []

    rater_prompt = settings.rater_prompt
    rate_prompt = settings.rate_prompt

    settings.max_new_tokens = settings.rate_tokens

    i = 0
    idx = 0
    while len(rates) < settings.probe_num:
        prompt = new_prompt(rater_prompt, responses[idx] + rate_prompt)
        i, rate = _generate_mean_rate(model, tokenizer, prompt, settings, i)
        rates.append(rate)
        idx += 1

    return rates
