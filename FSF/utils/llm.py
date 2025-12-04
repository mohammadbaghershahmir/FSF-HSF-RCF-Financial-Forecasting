import openai
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
        messages = [{"role": "user", "content": prompt}]
        completion = openai.ChatCompletion.create(model=self.model, messages=messages)
        response = completion.choices[0].message.content
        return response



class FastChatLLM:
    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, prompt):
        conv = get_conversation_template('vicuna-7b-1.5')
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        input = conv.get_prompt()

        input_ids = self.tokenizer([input]).input_ids
        output_ids = self.model.generate(
            torch.as_tensor(input_ids).to(self.model.device),
            do_sample=True,
            temperature=0.3,
            max_new_tokens=2048,
        )

        output_ids = output_ids[0][len(input_ids[0]) :]
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
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
            queries,
            do_sample=True,
            temperature=0.7,
            max_new_tokens=2048,
        )
        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        response = output[0]  # You can customize this if multiple outputs are generated
        return response
