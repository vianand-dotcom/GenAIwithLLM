from transformers import AutoModelForSeq2SeqLM, GenerationConfig
from pydataset import dataset
from loadpretrainedmodel import original_model, tokenizer
import torch
from zeroshotinfering import dash_line
instruct_model = AutoModelForSeq2SeqLM.from_pretrained(
    "./flan-dialogue-summary-checkpoint", torch_dtype=torch.bfloat16)
print(instruct_model)

index = 200
dialogue = dataset['test'][index]['dialogue']
human_baseline_summary = dataset['test'][index]['summary']

prompt = f"""
Summarize the following conversation.

{dialogue}

Summary:
"""

input_ids = tokenizer(prompt, return_tensors="pt").input_ids

original_model_outputs = original_model.generate(
    input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
original_model_text_output = tokenizer.decode(
    original_model_outputs[0], skip_special_tokens=True)

instruct_model_outputs = instruct_model.generate(
    input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
instruct_model_text_output = tokenizer.decode(
    instruct_model_outputs[0], skip_special_tokens=True)

print(dash_line)
print(f'BASELINE HUMAN SUMMARY:\n{human_baseline_summary}')
print(dash_line)
print(f'ORIGINAL MODEL:\n{original_model_text_output}')
print(dash_line)
print(f'INSTRUCT MODEL:\n{instruct_model_text_output}')
