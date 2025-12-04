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
                                    market=ticker,
                                    related_market="GOLD",
                                    examples=self.summarize_examples,
                                    tweets="\n".join(tweets))

            while len(self.tokenizer.encode(prompt)) > self.max_prompt_tok:
                tweets = tweets[:-1]
                prompt = self.summarize_prompt.format(
                                        market=ticker,
                                        related_market="GOLD",
                                        examples=self.summarize_examples,
                                        tweets="\n".join(tweets))

            raw = self.llm(prompt)
            summary = self._postprocess(raw)

        return summary

    def is_informative(self, summary):
        neg = r'.*[nN]o.*information.*|.*[nN]o.*facts.*|.*[nN]o.*mention.*|.*[nN]o.*tweets.*|.*do not contain.*'
        if not re.match(neg, summary):
            # Reject if excessive repetition remains
            if len(re.findall(r'(\b\w+\b)(?:\s+\1){3,}', summary, flags=re.IGNORECASE)) > 0:
                return False
            return True
        return False

    def _postprocess(self, text: str) -> str:
        # Collapse repeated tokens like "and and and"
        text = re.sub(r'(\band\b(?:\s+and\b)+)', 'and', text, flags=re.IGNORECASE)
        # Limit long runs of a single word repeated 3+ times
        text = re.sub(r'(\b\w+\b)(?:\s+\1){2,}', r'\1', text, flags=re.IGNORECASE)
        # Trim overly long bullet sections
        def trim_section(name: str, body: str) -> str:
            lines = [l for l in body.splitlines() if l.strip()]
            if len(lines) > 8:
                lines = lines[:8]
            return "\n".join(lines)
        parts = []
        current = []
        for line in text.splitlines():
            if line.strip().endswith(":"):
                if current:
                    parts.append("\n".join(current))
                    current = []
            current.append(line)
        if current:
            parts.append("\n".join(current))
        text = "\n\n".join(trim_section("sec", p) for p in parts)
        # Final length cap
        if len(text) > 2000:
            text = text[:2000]
        return text