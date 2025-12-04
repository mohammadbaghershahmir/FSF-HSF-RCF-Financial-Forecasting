from datasets import load_dataset
from transformers import AutoTokenizer


class RewardDataLoader(object):
    def __init__(self, dataset_name, train_subset, eval_subset, num_proc, tokenizer) -> None:
        super(RewardDataLoader, self).__init__()

        self.dataset_name = dataset_name
        self.train_subset = train_subset
        self.eval_subset = eval_subset
        self.num_proc = num_proc
        self.tokenizer = tokenizer  # استفاده از توکن‌ساز جدید Hugging Face

    # پیش‌پردازش داده‌ها برای تبدیل به جفت سوال و پاسخ‌ها
    def preprocess_function(self, examples):
        new_examples = {
            "input_ids_j": [],
            "attention_mask_j": [],
            "input_ids_k": [],
            "attention_mask_k": [],
            "preferred": []
        }

        for question, response_j, response_k in zip(examples["prompt"], examples["rejected"], examples["chosen"]):
            tokenized_j = self.tokenizer(
                "Question: " + question + "\n\nAnswer: " + response_j, truncation=True, padding="max_length", max_length=512)
            tokenized_k = self.tokenizer(
                "Question: " + question + "\n\nAnswer: " + response_k, truncation=True, padding="max_length", max_length=512)

            new_examples["input_ids_j"].append(tokenized_j["input_ids"])
            new_examples["attention_mask_j"].append(tokenized_j["attention_mask"])
            new_examples["input_ids_k"].append(tokenized_k["input_ids"])
            new_examples["attention_mask_k"].append(tokenized_k["attention_mask"])
            new_examples["preferred"].append(1)
        return new_examples

    def load_data(self):
        # بارگذاری داده‌ها
        train_dataset = load_dataset(self.dataset_name, split="train")
        if self.train_subset > 0:
            train_dataset = train_dataset.select(range(min(len(train_dataset), self.train_subset)))

        eval_dataset = load_dataset(self.dataset_name, split="train")
        if self.eval_subset > 0:
            eval_dataset = eval_dataset.select(range(min(len(eval_dataset), self.eval_subset)))

        original_columns = train_dataset.column_names

        # پیش‌پردازش داده‌ها
        print("train_dataset: ", len(train_dataset))
        train_dataset = train_dataset.map(
            self.preprocess_function, batched=True, num_proc=self.num_proc, remove_columns=original_columns
        )

        print("train_dataset: ", len(train_dataset))
        print("eval_dataset: ", len(eval_dataset))

        eval_dataset = eval_dataset.map(
            self.preprocess_function, batched=True, num_proc=self.num_proc, remove_columns=original_columns)

        print("eval_dataset: ", len(eval_dataset))

        return train_dataset, eval_dataset
