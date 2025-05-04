import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import accuracy_score

# Load small subset of IMDb
dataset = load_dataset("imdb")
small_dataset = dataset["train"].shuffle(seed=42).select(range(1000))  # Only 1k examples

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = small_dataset.map(tokenize_function, batched=True)

# Rename columns for Trainer compatibility
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Load model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)
# Evaluation metric
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

# Training arguments
training_args = TrainingArguments(
    output_dir="./bert-imdb-small",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    logging_steps=10,
    save_steps=100,
    save_total_limit=1,
    load_best_model_at_end=True,
    report_to="none",
    save_strategy="no"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,  # Just reuse since it's small
    compute_metrics=compute_metrics,
)

# Train
trainer.train()

# =============================
from sklearn.metrics import accuracy_score, classification_report
test_small = dataset["test"].shuffle(seed=42).select(range(200))

test_tokenized = test_small.map(tokenize_function, batched=True)
test_tokenized = test_tokenized.rename_column("label", "labels")
test_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Evaluate
eval_results = trainer.evaluate()
print("🔍 Evaluation Results:", eval_results)

# Predict on test
predictions_output = trainer.predict(test_tokenized)
preds = torch.argmax(torch.tensor(predictions_output.predictions), axis=-1)

# Classification report
print("\n📊 Classification Report on Test Set:")
print(classification_report(test_tokenized["labels"], preds, target_names=["negative", "positive"]))