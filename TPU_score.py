from transformers import AutoTokenizer, AutoModelForCausalLM
import math
import torch
import numpy as np

model_name = "meta-llama/Llama-3.1-8B"   
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

prompt = "Explain Token Probability-Based Uncertainty (TPU) in one paragraph."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=128,
    return_dict_in_generate=True,
    output_scores=True,
)

transition_scores = model.compute_transition_scores(
    outputs.sequences, outputs.scores, normalize_logits=True
)

input_len = inputs["input_ids"].shape[1]
generated_token_ids = outputs.sequences[:, input_len:]
generated_token_logprobs = transition_scores[0]  
avg_log = generated_token_logprobs.mean()
u_tpu = 1.0 - torch.exp(avg_log)
print("TPU =", u_tpu.item())
