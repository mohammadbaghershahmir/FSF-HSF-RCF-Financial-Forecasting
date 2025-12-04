from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os
from peft import PeftModel, PeftConfig

class VicunaModel:
	_instances = {}

	@classmethod
	def get_model(cls, model_name="lmsys/vicuna-7b-v1.5", quantization=True, device_map="auto"):
		"""
		Load model by name or local path. If the path is an adapter-only checkpoint (no config.json),
		first load the appropriate base model from the adapter's PEFT config and then apply the adapter.
		Also prefer a sibling tokenizer directory named 'dpo_tokenizer' when present.
		"""
		cache_key = f"{model_name}|{quantization}|{device_map}"
		if cache_key in cls._instances:
			return cls._instances[cache_key]

		print("Loading Vicuna Model...", model_name)
		quant_config = BitsAndBytesConfig(load_in_4bit=True) if quantization else None

		is_local_path = os.path.exists(model_name)
		is_adapter_only = False
		if is_local_path:
			config_path = os.path.join(model_name, "config.json")
			is_adapter_only = not os.path.isfile(config_path)

		if is_adapter_only:
			# Read adapter's PEFT config to get the correct base model path
			peft_cfg = PeftConfig.from_pretrained(model_name)
			base_name = peft_cfg.base_model_name_or_path or "lmsys/vicuna-7b-v1.5"
			base_model = AutoModelForCausalLM.from_pretrained(
				base_name,
				quantization_config=quant_config,
				device_map=device_map
			)
			model = PeftModel.from_pretrained(base_model, model_name)
			# Prefer a sibling tokenizer folder if it exists; else load tokenizer from base
			tokenizer_dir = os.path.join(os.path.dirname(model_name), "dpo_tokenizer")
			if os.path.isdir(tokenizer_dir):
				tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
			else:
				tokenizer = AutoTokenizer.from_pretrained(base_name)
		else:
			# Standard model load (full model directory or HF hub id)
			model = AutoModelForCausalLM.from_pretrained(
				model_name,
				quantization_config=quant_config,
				device_map=device_map
			)
			tokenizer = AutoTokenizer.from_pretrained(model_name)

		cls._instances[cache_key] = {
			"model": model,
			"tokenizer": tokenizer
		}
		print("Vicuna Model Loaded.")
		return cls._instances[cache_key] 