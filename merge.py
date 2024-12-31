#!/usr/bin/env python

import os
import shutil
import json
from peft import PeftModel
import torch
from transformers import MllamaForConditionalGeneration, BitsAndBytesConfig

BASE_MODEL = "./Llama-3.2-11B-Vision-Instruct-abliterated"
LORA_PATH = "checkpoint-7000"
OUTPUT_PATH = "./ludoviko"

os.makedirs(OUTPUT_PATH, exist_ok=True)

base_model = MllamaForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    torch_dtype=torch.float16,
    device_map="cuda"
)

model_to_merge = PeftModel.from_pretrained(base_model, LORA_PATH)
merged_model = model_to_merge.merge_and_unload()
merged_model.save_pretrained(OUTPUT_PATH)

shutil.copy2(os.path.join(BASE_MODEL, "tokenizer.json"), OUTPUT_PATH)
shutil.copy2(os.path.join(BASE_MODEL, "tokenizer_config.json"), OUTPUT_PATH)
