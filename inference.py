########################################################################################
#
# Inference Test for Distillation of a Reasoning Model
#
# Shows output from original base model + output from distilled model
#
# Luke Sheneman
# sheneman@uidaho.edu
# Institute for Interdisciplinary Data Sciences (IIDS)
#
# Sept 13, 2025
#
########################################################################################

from unsloth import FastLanguageModel
import torch
from transformers import TextStreamer, GenerationConfig

base_model_id = "google/gemma-3-1b-it"
adapter_path = "./gemma3_1b_live_distilled_qwen/epoch_3"
max_seq_length = 4096



# Load the base model
print(f"Loading base model: {base_model_id}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_model_id,
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)
print("Base model loaded successfully.")



# prepare test prompt
new_question = "Sarah has 3 boxes of crayons. Each box has 24 crayons. She gives 10 crayons to her friend. How many crayons does Sarah have left?"

messages = [
    {"role": "user", "content": f"Question: {new_question}"},
]

# Separate prompt formatting from tokenization for reliability ✨
# Step 1: Format the prompt into a single string.
formatted_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False, # <-- Do not tokenize here
    add_generation_prompt=True
)

# Step 2: Tokenize the formatted string. This reliably returns a dictionary.
inputs = tokenizer(
    [formatted_prompt], # Pass as a list for batching
    return_tensors="pt",
).to("cuda")

# Use a TextStreamer for a real-time output effect
text_streamer = TextStreamer(tokenizer, skip_prompt=True)

# Define shared generation parameters
generation_config = GenerationConfig(
    max_new_tokens=512,
    do_sample=True,
    temperature=0.1,
    top_p=0.9,
    pad_token_id=tokenizer.eos_token_id,
)



# Inference on just the un-distilled base model
print("\n" + "="*70)
print("                INFERENCE ON BASE MODEL (NO ADAPTER)")
print("="*70)
print(f"QUESTION:\n{new_question}\n")
print("MODEL'S ANSWER:")
print("-" * 70)

_ = model.generate(
    **inputs, # <-- This now works correctly
    streamer=text_streamer,
    generation_config=generation_config
)
print("\n" + "-"*70)


# Bind the adapter to the base model and try our distilled model
print("\n\n" + "="*70)
print("                APPLYING LORA ADAPTER")
print("="*70)
model.load_adapter(adapter_path)
print(f"LoRA adapter from '{adapter_path}' has been applied.")

print("\n\n" + "="*70)
print("      INFERENCE ON FINE-TUNED MODEL (WITH ADAPTER)")
print("="*70)
print(f"QUESTION:\n{new_question}\n")
print("MODEL'S ANSWER:")
print("-" * 70)

_ = model.generate(
    **inputs, # <-- This works correctly here as well
    streamer=text_streamer,
    generation_config=generation_config
)
print("\n" + "-"*70)

print("\n\nComparison complete.")
