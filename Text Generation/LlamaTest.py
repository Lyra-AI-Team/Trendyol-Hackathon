import streamlit as st
import os
import torch
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    pipeline,
    logging,
)

st.title("Lyra AI Text Generation")

user_input = st.text_input("Enter the informations:")

base_model = "meta-llama/Llama-3.2-1B" # https://huggingface.co/meta-llama/Llama-3.2-1B

hf_dataset = "ahmeterdempmk/Llama-E-Commerce-Fine-Tune-Data" # https://huggingface.co/ahmeterdempmk/Llama-E-Commerce-Fine-Tune-Data

dataset = load_dataset(hf_dataset, split="train")
model = AutoModelForCausalLM.from_pretrained (
    base_model,
    device_map={"": 0}
)
model.config.use_cache = False
model.config.pretraining_tp = 1
model.low_cpu_mem_usage=True
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
peft_params = LoraConfig (
    lora_alpha=16, # the scaling factor for the low-rank matrices
    lora_dropout=0.1, # the dropout probability of the LoRA layers
    r=64, # the dimension of the low-rank matrices
    bias="none",
    task_type="CAUSAL_LM", # the task to train for (sequence-to-sequence language modeling in this case)
)
training_params = TrainingArguments (
    output_dir="./LlamaResults",
    num_train_epochs=5, # One training epoch.
    per_device_train_batch_size=4, # Batch size per GPU for training.
    gradient_accumulation_steps=1, # This refers to the number of steps required to accumulate the gradients during the update process.
    optim="paged_adamw_32bit", # Model optimizer (AdamW optimizer).
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4, # Initial learning rate. (Llama 3.1 8B ile hesaplandı)
    weight_decay=0.001, # Weight decay is applied to all layers except bias/LayerNorm weights.
    fp16=False, # Disable fp16/bf16 training.
    bf16=False, # Disable fp16/bf16 training.
    max_grad_norm=0.3, # Gradient clipping.
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard"
)
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_params,
    dataset_text_field="input",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
)
train_output = trainer.train()

torch.cuda.empty_cache()

prompt = f"""
You are extracting product title and description from given text and rewriting the description and enhancing it when necessary.
Always give response in the user's input language.
Always answer in the given json format. Do not use any other keywords. Do not make up anything.
Explanations should contain at least three sentences each.

Json Format:
{{
"title": "<title of the product>",
"description": "<description of the product>"
}}

Examples:

Product Information: Rosehip Marmalade, keep it cold
Answer: {{"title": "Rosehip Marmalade", "description": "You should store this delicisious roseship marmelade in cold conditions. You can use it in your breakfasts and meals."}}

Product Information: Blackberry jam spoils in the heat
Answer: {{"title": "Blackberry Jam", "description": "Please store it in cold conditions. Recommended to be consumed at breakfast. Very sweet."}}

Now answer this:
Product Information: {user_input}"""

pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=10000)
result = pipe(f"Prompt: {prompt} \n Response:") # result = pipe(f"Prompt: {prompt} \n Response:")
generated_text = result[0]['generated_text']

st.write(generated_text)
