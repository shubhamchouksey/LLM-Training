from accelerate import Accelerator
from torch.optim import AdamW
from preprocessing_data import get_preprocessing_components
from transformers import AutoModelForSequenceClassification, get_scheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch
import evaluate


checkpoint = "bert-base-uncased"

accelerator = Accelerator()

tokenized_datasets, tokenizer, data_collator = get_preprocessing_components(checkpoint=checkpoint)

tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

optimizer = AdamW(model.parameters(), lr=3e-5)


train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)

train_dl, eval_dl, model, optimizer = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer
)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dl:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # Initialize the metric for the current epoch
    epoch_metric = evaluate.load("glue", "mrpc")

    # Evaluation phase
    model.eval()
    for batch in eval_dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        epoch_metric.add_batch(predictions=predictions, references=batch["labels"])

    print(f"\nEpoch {epoch + 1} evaluation:", epoch_metric.compute())