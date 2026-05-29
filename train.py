import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tokenizer import SimpleTokenizer
from model import SimpleGPT

# =========================
# Settings
# =========================

MAX_LENGTH = 64
BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 0.001

MODEL_FILE = "model.pth"

# =========================
# Build tokenizer
# =========================

tokenizer = SimpleTokenizer()
tokenizer.build_vocab("training_data.jsonl")

VOCAB_SIZE = len(tokenizer.word_to_id)

# =========================
# Dataset
# =========================

class ConversationDataset(Dataset):
    def __init__(self, jsonl_file, tokenizer):
        self.samples = []

        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)

                text = data["text"]

                tokens = tokenizer.encode(text)

                # Skip very short conversations
                if len(tokens) < 2:
                    continue

                # Trim
                tokens = tokens[:MAX_LENGTH]

                self.samples.append(tokens)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens = self.samples[idx]

        x = tokens[:-1]
        y = tokens[1:]

        # Padding
        while len(x) < MAX_LENGTH - 1:
            x.append(0)

        while len(y) < MAX_LENGTH - 1:
            y.append(0)

        return (
            torch.tensor(x),
            torch.tensor(y)
        )

dataset = ConversationDataset(
    "training_data.jsonl",
    tokenizer
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# =========================
# Create model
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = SimpleGPT(
    vocab_size=VOCAB_SIZE,
    embed_size=128,
    num_heads=4,
    num_layers=2,
    max_length=MAX_LENGTH
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

criterion = nn.CrossEntropyLoss(ignore_index=0)

# =========================
# Training loop
# =========================

print("Starting training...")
print("Device:", device)

for epoch in range(EPOCHS):

    total_loss = 0

    for x, y in loader:

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        print("Max token:", x.max().item())
        print("Vocab size:", VOCAB_SIZE)

        outputs = model(x)

        loss = criterion(
            outputs.reshape(-1, VOCAB_SIZE),
            y.reshape(-1)
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)

    print(
        f"Epoch {epoch+1}/{EPOCHS} "
        f"Loss: {avg_loss:.4f}"
    )

# =========================
# Save model
# =========================

torch.save({
    "model_state": model.state_dict(),
    "vocab": tokenizer.word_to_id
}, MODEL_FILE)

print("Model saved to", MODEL_FILE)
