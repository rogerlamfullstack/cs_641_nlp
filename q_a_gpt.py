import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
from sklearn.metrics import accuracy_score

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# Load small SQuAD v1.1 subset
dataset = load_dataset("squad")
train_data = dataset["train"].select(range(1000))
test_data = dataset["validation"].select(range(100))

# Format as prompt: Context + Question → Answer
def format_example(example):
    prompt = f"Context: {example['context']}\nQuestion: {example['question']}\nAnswer:"
    label = " " + example["answers"]["text"][0]  # Take the first annotated answer
    return {"text": prompt + label}

train_data = train_data.map(format_example)
test_data = test_data.map(format_example)

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

# Tokenize
def tokenize_fn(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=256)

train_data = train_data.map(tokenize_fn, batched=True)
test_data = test_data.map(tokenize_fn, batched=True)

train_data.set_format(type="torch", columns=["input_ids", "attention_mask"])
test_data.set_format(type="torch", columns=["input_ids", "attention_mask"])

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training args
args = TrainingArguments(
    output_dir="./gpt2-squad",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=2,
    save_strategy="no",
    load_best_model_at_end=True,
    logging_steps=20,
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_data,
    eval_dataset=test_data,
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Evaluation loss
print("\n🔍 Evaluation:")
print(trainer.evaluate())

# Predict function
def predict_answer(context, question):
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=30)
    return tokenizer.decode(output[0][input_ids.shape[-1]:]).strip()

# Try on test sample
sample = test_data[0]
context = dataset["validation"][0]["context"]
question = dataset["validation"][0]["question"]
true_answer = dataset["validation"][0]["answers"]["text"][0]

predicted_answer = predict_answer(context, question)

print("\nSample QA:")
print(f"Q: {question}")
print(f"A (true): {true_answer}")
print(f"A (pred): {predicted_answer}")
