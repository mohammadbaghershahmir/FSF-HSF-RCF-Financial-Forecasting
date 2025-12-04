from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
import sys
import torch
import transformers
import warnings
from datasets import load_dataset
from predict_module import sft_dataloader

def supervised_finetune(args):
    MICRO_BATCH_SIZE = 1  # کاهش برای مصرف کمتر VRAM
    BATCH_SIZE = 4
    EPOCHS = 2
    LEARNING_RATE = 3e-4
    CUTOFF_LEN = 128
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    VAL_PCT = 0.1
    TARGET_MODULES = ["q_proj", "v_proj"]
    DATA_PATH = args.data_path
    OUTPUT_DIR = args.output_path

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else "auto"
    GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
    if ddp:
        GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size

    print(f"Loading model from {args.model_path}")
    torch.cuda.empty_cache()
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        load_in_8bit_fp32_cpu_offload=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        offload_folder="offload",  # در صورت نیاز به offloading
        offload_state_dict=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=True,
        add_eos_token=True
    )
    tokenizer.pad_token_id = 0

    model = prepare_model_for_int8_training(model)
    model.gradient_checkpointing_enable()

    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)

    data = load_dataset("json", data_files=DATA_PATH)
    val_set_size = int(VAL_PCT * len(data["train"]))

    MAX_STEPS = int(max((len(data["train"]) - val_set_size) // BATCH_SIZE * EPOCHS, EPOCHS))
    if args.resume_from_supervised_checkpoint:
        checkpoint_name = os.path.join(args.resume_from_supervised_checkpoint, "pytorch_model.bin")
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(args.resume_from_supervised_checkpoint, "adapter_model.bin")
        if os.path.exists(checkpoint_name):
            print(f"Resuming from checkpoint: {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)

    model.print_trainable_parameters()

    dataloader = sft_dataloader.SFTDataLoader(data, CUTOFF_LEN, val_set_size, tokenizer)
    train_data, val_data = dataloader.load_data()

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=MICRO_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            warmup_steps=10,
            num_train_epochs=EPOCHS,
            max_steps=MAX_STEPS,
            learning_rate=LEARNING_RATE,
            fp16=True,
            logging_steps=20,
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=args.eval_steps if val_set_size > 0 else None,
            save_steps=args.save_steps,
            output_dir=OUTPUT_DIR,
            save_total_limit=1,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            report_to="wandb" if args.wandb else [],
            ignore_data_skip=args.ignore_data_skip,
            gradient_checkpointing=True,
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=args.resume_from_supervised_checkpoint)
    model.save_pretrained(OUTPUT_DIR)
