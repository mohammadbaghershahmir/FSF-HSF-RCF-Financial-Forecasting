from utils.llm import FastChatLLM
from utils.prompts import SUMMARIZE_INSTRUCTION
from utils.fewshots import SUMMARIZE_EXAMPLES
import tiktoken
import re
from model_loader import VicunaModel
import torch
model_data = VicunaModel.get_model(
    model_name="lmsys/vicuna-7b-v1.5",
    quantization="bnb_4bit",          
    device_map="auto"
)
class Summarizer:
    RESERVED_GEN = 256
    def __init__(self):
        self.summarize_prompt = SUMMARIZE_INSTRUCTION
        self.summarize_examples = SUMMARIZE_EXAMPLES
        self.tokenizer = model_data["tokenizer"]  # تعریف tokenizer از model_data
        self.llm = FastChatLLM(model=model_data["model"], tokenizer=model_data["tokenizer"])
        for m in self.llm.model.modules():
            if hasattr(m, "compute_dtype"):
                m.compute_dtype = torch.float16

        self.max_prompt_tok = self.tokenizer.model_max_length - self.RESERVED_GEN
    def get_summary(self, ticker, tweets):
        summary = None
        if tweets != []:
            prompt = self.summarize_prompt.format(
                                    ticker = ticker,
                                    examples = self.summarize_examples,
                                    tweets = "\n".join(tweets))

            while len(self.tokenizer.encode(prompt)) > self.max_prompt_tok:
                tweets = tweets[:-1]
                prompt = self.summarize_prompt.format(
                                        ticker = ticker,
                                        examples = self.summarize_examples,
                                        tweets = "\n".join(tweets))

            summary = self.llm(prompt)

        return summary

    def is_informative(self, summary):
        neg = r'.*[nN]o.*information.*|.*[nN]o.*facts.*|.*[nN]o.*mention.*|.*[nN]o.*tweets.*|.*do not contain.*'
        return not re.match(neg, summary)