########################################################################################
#
# Live Response-Based Distillation of a Reasoning Model
#
# Luke Sheneman
# sheneman@uidaho.edu
# Institute for Interdisciplinary Data Sciences (IIDS)
# 
# Sept 13, 2025
#
# Teacher: Qwen/Qwen3-4B-Thinking-2507
# Student: google/gemma-3-1b-it
#
# Domain:  Grade School Mathematics
#
########################################################################################

from unsloth import FastLanguageModel
import torch
import gc
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from tqdm import tqdm
import os # 

# ======================================================================================
# CONFIGURATION
# ======================================================================================
teacher_model_id = "Qwen/Qwen3-4B-Thinking-2507"
student_model_id = "google/gemma-3-1b-it"
output_dir = "gemma3_1b_live_distilled_qwen" # <-- Define a base output directory

max_seq_length = 4096
num_epochs = 4
live_batch_size = 128



# ======================================================================================
# LOAD MODELS (Teacher and Student)
# ======================================================================================
print(f"Loading teacher model: {teacher_model_id}")
teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_id)
teacher_model = AutoModelForCausalLM.from_pretrained(
    teacher_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
print("Teacher model loaded.")

print(f"Loading student model with Unsloth: {student_model_id}")
student_model, student_tokenizer = FastLanguageModel.from_pretrained(
    model_name=student_model_id,
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True,
)

student_model = FastLanguageModel.get_peft_model(
    student_model,
    r=16, lora_alpha=16, lora_dropout=0, bias="none",
    use_gradient_checkpointing=True, random_state=3407,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_up_proj", "down_proj"],
)
print("Student model loaded and prepared for training.")




# ======================================================================================
# LOAD THE SOURCE DATASET
# ======================================================================================
gsm8k_dataset = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=42)






# ======================================================================================
# THE LIVE DISTILLATION LOOP
# ======================================================================================
print("\n🚀 Starting live distillation process...")

for epoch in range(num_epochs):
    print(f"\n--- Starting Epoch {epoch + 1} of {num_epochs} ---")

    progress_bar = tqdm(range(0, len(gsm8k_dataset), live_batch_size), desc=f"Epoch {epoch+1}")
    for i in progress_bar:

        chunk_indices = range(i, min(i + live_batch_size, len(gsm8k_dataset)))
        live_batch = gsm8k_dataset.select(chunk_indices)

        teacher_prompts = []
        for example in live_batch:
            # Explicit one-shot system directive
            system_prompt = (
                "You are a helpful math assistant. You will be given a math word problem. "
                "Your response must start with a \"<think>\" tag. Inside the tag, provide your step-by-step reasoning. "
                "After the closing </think> tag, provide the final answer."
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": example["question"]},
            ]

            # Build chat prompt and PREFILL the assistant with "<think>"
            prompt = teacher_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt += "<think>"  # <-- guarantees the very first characters are "<think>"
            teacher_prompts.append(prompt)

        inputs = teacher_tokenizer(
            teacher_prompts, return_tensors="pt", padding=True
        ).to(teacher_model.device)

        # Predictable structure
        outputs = teacher_model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            pad_token_id=teacher_tokenizer.eos_token_id,
        )

        # Keep skip_special_tokens=True, but we prefilled "<think>" as plain text,
        # so it won't be stripped. Still, normalize just in case.
        decoded_answers = teacher_tokenizer.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        # Normalize to ensure the response fed to Gemma starts with "<think>"
        distilled_data_chunk = []
        for ex, ans in zip(live_batch, decoded_answers):
            fixed = ans.lstrip()
            if not fixed.startswith("<think>"):
                fixed = "<think>" + fixed
            distilled_data_chunk.append({
                "question": ex["question"],
                "qwen3_answer": fixed
            })

        # Simple demo to see training data example during training
        if i == 0: # <-- Only show demo for the first batch of each epoch
            print("\n" + "=" * 70)
            print("               EXAMPLE FROM ONE-SHOT PROMPT")
            print("=" * 70)
            print(f"----\n[QUESTION]:\n{distilled_data_chunk[0]['question']}\n")
            print(f"[QWEN-GENERATED OUTPUT (normalized)]:\n{distilled_data_chunk[0]['qwen3_answer']}")
            print("=" * 70 + "\n")

        # Create a dataset for Gemma training
        training_dataset = Dataset.from_list(distilled_data_chunk)

        def format_student_prompt(example):
            # Train Gemma on the entire structured output, as assistant
            messages = [
                {"role": "user", "content": f"Question: {example['question']}"},
                {"role": "assistant", "content": f"{example['qwen3_answer']}"},
            ]
            text = student_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            return {"text": text}

        formatted_training_dataset = training_dataset.map(format_student_prompt, num_proc=4)



        # Demo - What Gemma is trained on (Qwen3 output) for the first example in this batch
        if i == 0: 
            gemma_training_preview = formatted_training_dataset[0]["text"]
            print("\n" + "-" * 70)
            print("WHAT GEMMA SEES (training text for first sample):")
            print("-" * 70)
            print(gemma_training_preview)
            print("-" * 70 + "\n")

        trainer = SFTTrainer(
            model=student_model,
            tokenizer=student_tokenizer,
            train_dataset=formatted_training_dataset,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            packing=False,
            args=TrainingArguments(
                per_device_train_batch_size=4,
                gradient_accumulation_steps=8,
                num_train_epochs=1,
                warmup_steps=1,
                learning_rate=2e-4,
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=1,
                optim="adamw_8bit",
                output_dir="outputs", # Temporary trainer-specific output
                report_to="none",
            ),
        )

        train_result = trainer.train()
        loss = train_result.training_loss
        progress_bar.set_postfix({"loss": f"{loss:.4f}"})

        # Cleanup
        del trainer, training_dataset, formatted_training_dataset, inputs, outputs
        gc.collect()
        torch.cuda.empty_cache()

    # Save checkpoint at the end of each epoch
    checkpoint_dir = os.path.join(output_dir, f"epoch_{epoch + 1}")
    os.makedirs(checkpoint_dir, exist_ok=True) # Ensure the directory exists
    student_model.save_pretrained(checkpoint_dir)
    print(f"Checkpoint for epoch {epoch + 1} saved to {checkpoint_dir}")


print(f"\n\n Live distillation and fine-tuning complete for {num_epochs} epochs!")

# Save the final model/adapters to a 'final' directory
final_dir = os.path.join(output_dir, "final")
student_model.save_pretrained(final_dir)
print(f"Final model adapters saved to {final_dir}")
