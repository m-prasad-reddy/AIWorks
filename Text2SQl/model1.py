from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import torch
import json

# Load Pretraining Data
with open("text_sql_pairs.json", "r") as f:
    dataset = json.load(f)

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Tokenize Data
input_texts = [f"translate English to SQL: {d['text']}" for d in dataset]
target_texts = [d['sql'] for d in dataset]

inputs = tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt")
targets = tokenizer(target_texts, padding=True, truncation=True, return_tensors="pt")

# Prepare Dataset
class SQLDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": self.inputs["input_ids"][idx],
            "attention_mask": self.inputs["attention_mask"][idx],
            "labels": self.targets["input_ids"][idx]
        }

dataset = SQLDataset(inputs, targets)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./sql_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir="./logs"
)

trainer = Trainer(
    model=model,
    args=training_args,
     eval_dataset=dataset,
    train_dataset=dataset
)

# Train Model
trainer.train()
model.save_pretrained("sql_model")
tokenizer.save_pretrained("sql_model")

print("Training completed. Model saved.")
