from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

def get_preprocessing_components(dataset_name="glue", subset_name="mrpc", checkpoint="bert-base-uncased"):
    raw_datasets = load_dataset(dataset_name, subset_name)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def tokenize_function(example):
        return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return tokenized_datasets, tokenizer, data_collator

