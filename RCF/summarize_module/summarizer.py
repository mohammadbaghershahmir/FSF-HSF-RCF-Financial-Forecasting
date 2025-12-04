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
    RESERVED_GEN = 512
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

            # Only truncate if absolutely necessary, and do it more intelligently
            if len(self.tokenizer.encode(prompt)) > self.max_prompt_tok:
                # Try to keep more tweets by reducing examples first
                shorter_examples = self.summarize_examples[:len(self.summarize_examples)//2]
                prompt = self.summarize_prompt.format(
                                        market=ticker,
                                        related_market="GOLD",
                                        examples=shorter_examples,
                                        tweets="\n".join(tweets))
                
                # If still too long, truncate tweets more conservatively
                if len(self.tokenizer.encode(prompt)) > self.max_prompt_tok:
                    # Keep 80% of tweets instead of removing one by one
                    keep_ratio = 0.8
                    tweets_to_keep = int(len(tweets) * keep_ratio)
                    tweets = tweets[:tweets_to_keep]
                    prompt = self.summarize_prompt.format(
                                            market=ticker,
                                            related_market="GOLD",
                                            examples=shorter_examples,
                                            tweets="\n".join(tweets))

            raw = self.llm(prompt)
            summary = self._postprocess(raw)

        return summary

    def is_informative(self, summary):
        # More lenient check - only reject if explicitly says no information
        neg = r'.*[nN]o.*information.*|.*[nN]o.*facts.*|.*[nN]o.*mention.*|.*[nN]o.*tweets.*|.*do not contain.*|.*cannot.*analyze.*|.*unable.*to.*'
        if re.match(neg, summary, re.IGNORECASE):
            return False
        
        # Check for minimum meaningful content
        if len(summary.strip()) < 50:
            return False
            
        # Reject if excessive repetition remains
        if len(re.findall(r'(\b\w+\b)(?:\s+\1){3,}', summary, flags=re.IGNORECASE)) > 0:
            return False
        return True

    def _postprocess(self, text: str) -> str:
        # Collapse repeated tokens like "and and and"
        text = re.sub(r'(\band\b(?:\s+and\b)+)', 'and', text, flags=re.IGNORECASE)
        # Limit long runs of a single word repeated 3+ times
        text = re.sub(r'(\b\w+\b)(?:\s+\1){2,}', r'\1', text, flags=re.IGNORECASE)
        # Remove/merge repeated sentences
        sents = re.split(r'(?<=[.!?])\s+', text.strip())
        seen = set()
        clean_sents = []
        for s in sents:
            ss = s.strip()
            if not ss:
                continue
            key = ss.lower()
            if key in seen:
                continue
            seen.add(key)
            clean_sents.append(ss)
        # If too few unique sentences or dominated by repetition, return neutral minimal message
        if len(clean_sents) == 0:
            return "No meaningful market news found for the day."
        # Keep only the first informative sentence and cap length
        first = clean_sents[0]
        # Trim to ~30 words
        words = first.split()
        if len(words) > 30:
            first = " ".join(words[:30])
        # Ensure punctuation
        if not first.endswith(('.', '!', '?')):
            first = first.rstrip('.') + '.'
        return first