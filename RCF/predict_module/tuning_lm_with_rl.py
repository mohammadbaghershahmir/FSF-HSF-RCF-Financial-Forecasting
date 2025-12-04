import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, TaskType, get_peft_model

def preprocess_function(examples, tokenizer):
    return {
        "queries": examples["prompt"],
        "responses_a": examples["rejected"],
        "responses_b": examples["chosen"],
    }

def tuning_lm_with_dpo(args):
    print("Loading model and tokenizer...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int8_enable_fp32_cpu_offload=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.rl_base_model,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
    )

    model = get_peft_model(model, peft_config)

    print("Loading and preprocessing dataset...")
    dataset = load_dataset("json", data_files={"train": "datasets/comparison_data_multi_model.json"}, split="train")
    dataset = dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)

    dpo_config = DPOConfig(
        learning_rate=args.rl_learning_rate,
        gradient_accumulation_steps=args.rl_gradient_accumulation_steps,
        output_dir=args.output_dir,
        logging_steps=10,
        per_device_train_batch_size=1
    )

    print("Initializing DPOTrainer...")
    dpo_trainer = DPOTrainer(
        args=dpo_config,
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
    )

    print("Starting training with DPO...")
    dpo_trainer.train()

    print("Saving the fine-tuned model...")
    model.save_pretrained(os.path.join(args.output_dir, "dpo_model"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "dpo_tokenizer"))
    print("Training completed and model saved.")
