from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class VicunaModel:
    _instance = None

    @classmethod
    def get_model(cls, model_name="lmsys/vicuna-7b-v1.5", quantization=True, device_map="auto"):
        """
        """
        if cls._instance is None:
            print("Loading Vicuna Model...")
            
            # تنظیمات Quantization
            quant_config = BitsAndBytesConfig(load_in_4bit=True) if quantization else None
            
            # بارگذاری مدل با تنظیمات Quantization و توزیع خودکار
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quant_config,
                device_map=device_map
            )
            
            # بارگذاری tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            cls._instance = {
                "model": model,
                "tokenizer": tokenizer
            }
            print("Vicuna Model Loaded.")
        
        return cls._instance
