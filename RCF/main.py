from exp.exp_multi_model import Exp_MultiModel
import argparse
import torch
import numpy as np
import random

# Fix seed for reproducibility
fix_seed = 100
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# Argument parser
parser = argparse.ArgumentParser(description='Training and evaluation for DPO-based stock prediction')

# Data loading settings
parser.add_argument("--price_dir", type=str, default="data/price/preprocessed/")
parser.add_argument("--tweet_dir", type=str, default="data/sample_tweet/raw/")
parser.add_argument("--seq_len", type=int, default=1)

# Supervised fine-tuning settings
parser.add_argument("--wandb", action="store_true", default=False)
parser.add_argument("--data_path", type=str, default="./datasets/merge_sample_multi_model.json")
parser.add_argument("--output_path", type=str, default="./saved_models/lora-Tiny-Vicuna")
parser.add_argument("--model_path", type=str, default="Jiayi-Pan/Tiny-Vicuna-1B")
parser.add_argument("--eval_steps", type=int, default=100)
parser.add_argument("--save_steps", type=int, default=100)
parser.add_argument("--resume_from_supervised_checkpoint", type=str, default=None)
parser.add_argument("--ignore_data_skip", type=str, default="False")

# Reflection and dataset generation
parser.add_argument("--num_reflect_trials", type=int, default=4)
parser.add_argument("--datasets_dir", type=str, default="./datasets/")
parser.add_argument('--local_rank', type=int, default=0, help="Used for multi-GPU training")
parser.add_argument('--deepspeed', type=str, default=None, help="Path to deepspeed config if using deepspeed.")
parser.add_argument('--output_dir', type=str, default="./saved_models/tuning_tiny_vicuna_dpo_checkpoints/", help="directory to save the model")
# DPO settings
parser.add_argument('--rl_base_model', type=str, default="./saved_models/lora-Tiny-Vicuna-adapter-merged", help="Base model for DPO training")
parser.add_argument('--tokenizer_name', type=str, default="Jiayi-Pan/Tiny-Vicuna-1B", help="Tokenizer for the base model")
parser.add_argument('--rl_learning_rate', type=float, default=1.4e-5, help="Learning rate for DPO training")
parser.add_argument('--dpo_epochs', type=int, default=20, help="Number of epochs for DPO training.")
parser.add_argument('--dpo_scale', type=float, default=1.0, help="Scale for DPO loss.")
parser.add_argument('--output_max_length', type=int, default=128, help="Maximum output length for generation")
parser.add_argument('--preferred', type=str, default="chosen", help="Preferred completion (default: chosen)")
parser.add_argument('--log_with', type=str, default=None, help="Use 'wandb' for logging.")
parser.add_argument('--early_stopping', type=bool, default=True, help="Whether to use early stopping.")
parser.add_argument('--seed', type=int, default=0, help="Seed for random number generators")
parser.add_argument('--mini_batch_size', type=int, default=1, help="the DPO minibatch size")
parser.add_argument('--rl_gradient_accumulation_steps', type=int, default=1, help="the number of gradient accumulation steps")
parser.add_argument('--batch_size', type=int, default=1, help="the batch size")
# Multi-model settings
parser.add_argument("--num_models", type=int, default=3, help="Number of models in ensemble (must be odd)")

# Evaluation settings
parser.add_argument("--num_shots", type=int, default=4)
parser.add_argument("--save_dir", type=str, default="results/")

# Parse arguments
args = parser.parse_args()
print('Args in experiment:')
print(args)

# Initialize the experiment model
print("Initializing Experiment Model...")
exp_model = Exp_MultiModel(args)
print("Experiment Model Initialized.")

# Start training
print("Starting Training with Multi-Model Architecture...")
#exp_model.train()
print("Training Completed.")

# Auto-start testing (single-agent DPO-only)
print("Starting Testing (DPO-only)...")
exp_model.test()
print("Testing Completed.")#
