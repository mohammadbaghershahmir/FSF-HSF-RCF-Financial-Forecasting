try:
    import openai  # optional
except Exception:
    openai = None
from tenacity import (
    retry,
    stop_after_attempt, # type: ignore
    wait_random_exponential, # type: ignore
)
from fastchat.model import get_conversation_template
import torch


class OpenAILLM:
    def __init__(self):
        self.model = "gpt-3.5-turbo"
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def __call__(self, prompt):
        if openai is None:
            raise RuntimeError("OpenAI SDK not installed. Install 'openai' or avoid using OpenAILLM.")
        messages = [{"role": "user", "content": prompt}]
        completion = openai.ChatCompletion.create(model=self.model, messages=messages)
        response = completion.choices[0].message.content
        return response



class FastChatLLM:
    def __init__(self, model=None, tokenizer=None, *, do_sample: bool = False, temperature: float = 1.0, max_new_tokens: int = 768, top_p: float = 1.0, top_k: int = 50, repetition_penalty: float = 1.0, use_raw_prompt: bool = True):
        self.model = model
        self.tokenizer = tokenizer
        self.do_sample = do_sample
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.use_raw_prompt = use_raw_prompt

    def __call__(self, prompt):
        if self.use_raw_prompt:
            input = prompt
        else:
            conv = get_conversation_template('vicuna-7b-1.5')
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            input = conv.get_prompt()

        # Encode with truncation to model context
        max_ctx = getattr(getattr(self.model, 'config', None), 'max_position_embeddings', 8192)
        enc = self.tokenizer(
            input,
            return_tensors='pt',
            truncation=True,
            max_length=max_ctx
        )
        input_ids = enc.input_ids.to(self.model.device)

        # Generate with keyword args; fallback to base_model.generate if needed
        try:
            output_ids = self.model.generate(
                input_ids=input_ids,
                do_sample=self.do_sample,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                top_p=self.top_p,
                top_k=self.top_k,
                repetition_penalty=self.repetition_penalty,
            )
        except TypeError:
            base = getattr(self.model, 'base_model', self.model)
            output_ids = base.generate(
                input_ids=input_ids,
                do_sample=self.do_sample,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                top_p=self.top_p,
                top_k=self.top_k,
                repetition_penalty=self.repetition_penalty,
        )

        gen_ids = output_ids[0][input_ids.shape[1]:]
        response = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        return response


class NShotLLM:
    def __init__(self, model=None, tokenizer=None, num_shots=4):
        """
        Initializes the N-shot language model.

        Args:
            model: The fine-tuned DPO model.
            tokenizer: Tokenizer associated with the model.
            num_shots: Number of shots to include in the prompt.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.num_shots = num_shots
    def __call__(self, prompt):
        # Get the device of the model
        device = next(self.model.parameters()).device
    
        # Encode the prompt and move it to the model's device
        query = self.tokenizer.encode(prompt, return_tensors="pt").to(device)
    
        queries = query.repeat((self.num_shots, 1))
        output_ids = self.model.generate(
            input_ids=queries,
            do_sample=True,
            temperature=0.75,
            max_new_tokens=512,
        )
        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        response = output[0]  # You can customize this if multiple outputs are generated
        return response
