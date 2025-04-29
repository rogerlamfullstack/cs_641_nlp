# Install requirements first if needed
# pip install transformers datasets torch scikit-learn

from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
import torch

# 1. Load the TREC dataset
dataset = load_dataset("trec", trust_remote_code=True)

# 2. Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 3. Preprocessing: Tokenize the data
def preprocess_function(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length', 
        max_length=64   
    )


tokenized_dataset = dataset.map(preprocess_function, batched=True)

# 4. Prepare dataset format for PyTorch
tokenized_dataset = tokenized_dataset.rename_column("coarse_label", "labels")

tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# 5. Load pre-trained BERT model (for classification)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=6)  # TREC has 6 classes

# 6. Define metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

# 7. TrainingArguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="no"
)

# 8. Trainer API
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
)

# 9. Train the model
trainer.train()

# 10. Evaluate the model
results = trainer.evaluate()
print(f"Test Accuracy: {results['eval_accuracy']:.4f}")

# 11. Predict a sample
sample = "What is the NLP?"
inputs = tokenizer(sample, return_tensors="pt")
outputs = model(**inputs)
predicted_class = outputs.logits.argmax().item()
print(f"Predicted class ID: {predicted_class}")