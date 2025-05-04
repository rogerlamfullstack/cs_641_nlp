from transformers import pipeline
import torch
from transformers import BertTokenizer, BertForSequenceClassification

utterances = [
    "Hey everyone!",
    "Hi Alice!",
    "How do I install PyTorch?",
    "Just use pip install torch",
    "Thanks!",
    "Are we meeting at 5pm?",
    "Yes, in Room 101",
    "Great, see you there.",
    "Any idea about CUDA version?",
    "Check nvidia-smi in terminal."
]

# Load pretrained BERT model for sentence-pair classification
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Example: 10 interleaved chat utterances
utterances = [
    "Hey everyone!",
    "Hi Alice!",
    "How do I install PyTorch?",
    "Just use pip install torch",
    "Thanks!",
    "Are we meeting at 5pm?",
    "Yes, in Room 101",
    "Great, see you there.",
    "Any idea about CUDA version?",
    "Check nvidia-smi in terminal."
]

def predict_reply_links(utterances, top_k=1):
    links = []

    for i in range(len(utterances)):
        u_i = utterances[i]
        candidates = []
        for j in range(i):
            u_j = utterances[j]
            encoded = tokenizer(u_i, u_j, return_tensors="pt", padding=True, truncation=True, max_length=128)
            encoded = {k: v.to(device) for k, v in encoded.items()}

            with torch.no_grad():
                output = model(**encoded)
                prob = torch.softmax(output.logits, dim=1)[0, 1].item()  # probability of "reply"
                candidates.append((j, prob))

        # Get top-k reply candidates
        if candidates:
            best_j, best_prob = max(candidates, key=lambda x: x[1])
            if best_prob > 0.5:
                links.append((i, best_j, round(best_prob, 2)))
            else:
                links.append((i, None, round(best_prob, 2)))  # no reply
        else:
            links.append((i, None, 0.0))  # first utterance

    return links

# Run inference
reply_links = predict_reply_links(utterances)

# Display reply predictions
print("\n ===============Predicted Reply Links:")
for i, j, score in reply_links:
    u_i = utterances[i]
    reply_to = f"→ [{j}] {utterances[j]}" if j is not None else "→ None"
    print(f"[{i}] {u_i} {reply_to} (confidence={score})")
