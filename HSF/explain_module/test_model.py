from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import os

class TestModel:
    """Class for loading fine-tuned models for testing"""
    
    def __init__(self, model_path="./saved_models/sep_model"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        
        # Alternative paths for different model types
        self.dpo_model_path = "./saved_models/tuning_tiny_vicuna_dpo_checkpoints/dpo_model"
        self.dpo_tokenizer_path = "./saved_models/tuning_tiny_vicuna_dpo_checkpoints/dpo_tokenizer"
        self.merged_model_path = "./saved_models/lora-Tiny-Vicuna-adapter-merged"
        
    def load_model(self):
        """Load the fine-tuned model for testing"""
        if self.model is None:
            print(f"Loading Test Model...")
            
            try:
                # First try: Load from DPO model (complete with tokenizer)
                if os.path.exists(self.dpo_model_path) and os.path.exists(self.dpo_tokenizer_path):
                    print("Loading from DPO model (complete with tokenizer)...")
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        llm_int8_enable_fp32_cpu_offload=True
                    )
                    
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.dpo_model_path,
                        device_map="auto",
                        quantization_config=bnb_config,
                        torch_dtype=torch.float16
                    )
                    
                    self.tokenizer = AutoTokenizer.from_pretrained(self.dpo_tokenizer_path)
                    print("✅ Test Model Loaded Successfully from DPO model")
                # Second try: Load from sep_model (model only, use base tokenizer)
                elif os.path.exists(os.path.join(self.model_path, "model.safetensors")):
                    print("Loading from sep_model (using base model tokenizer)...")
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        llm_int8_enable_fp32_cpu_offload=True
                    )
                    
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        device_map="auto",
                        quantization_config=bnb_config,
                        torch_dtype=torch.float16
                    )
                    
                    # Use base model tokenizer since sep_model doesn't have one
                    print("⚠️  sep_model has no tokenizer, using base model tokenizer...")
                    from model_loader import VicunaModel
                    base_model_data = VicunaModel.get_model()
                    self.tokenizer = base_model_data["tokenizer"]
                    print("✅ Test Model Loaded Successfully from sep_model with base tokenizer")
                # Third try: Load from merged LoRA model
                elif os.path.exists(self.merged_model_path):
                    print("Loading from merged LoRA model...")
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        llm_int8_enable_fp32_cpu_offload=True
                    )
                    
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.merged_model_path,
                        device_map="auto",
                        quantization_config=bnb_config,
                        torch_dtype=torch.float16
                    )
                    
                    # Try to load tokenizer from merged model
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(self.merged_model_path)
                        print("✅ Test Model Loaded Successfully from merged LoRA model")
                    except:
                        # If no tokenizer, use base model tokenizer
                        print("⚠️  No tokenizer found, using base model tokenizer...")
                        from model_loader import VicunaModel
                        base_model_data = VicunaModel.get_model()
                        self.tokenizer = base_model_data["tokenizer"]
                        print("✅ Using base model tokenizer")
                        
                else:
                    raise Exception("No suitable fine-tuned model found")
                
            except Exception as e:
                print(f"❌ Error loading test model: {e}")
                print("🔄 Falling back to base Vicuna model...")
                # Fallback to base model if fine-tuned model fails
                from model_loader import VicunaModel
                model_data = VicunaModel.get_model()
                self.model = model_data["model"]
                self.tokenizer = model_data["tokenizer"]
        
        return {
            "model": self.model,
            "tokenizer": self.tokenizer
        }
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if self.model is not None:
            return {
                "model_path": self.model_path,
                "model_type": type(self.model).__name__,
                "device": next(self.model.parameters()).device if hasattr(self.model, 'parameters') else "unknown",
                "has_tokenizer": self.tokenizer is not None
            }
        return {"status": "Model not loaded"} 