import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from tokenizer import SimpleTokenizer
from model import SimpleGPT

# =========================
# Settings
# =========================

MAX_LENGTH = 256
BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 0.001

EMBED_SIZE = 128
NUM_HEADS = 4
NUM_LAYERS = 2

MODEL_FILE = "model.pth"

# =========================
# Build tokenizer
# =========================

DATA_DIR = Path("training_data")

TRAINING_FILES = sorted(DATA_DIR.glob("*.jsonl"))

if not TRAINING_FILES:
    raise FileNotFoundError(
        f"No .jsonl training files found in {DATA_DIR.resolve()}"
    )

print("Training files:")
for file in TRAINING_FILES:
    print(" -", file)

tokenizer = SimpleTokenizer()
tokenizer.build_vocab(TRAINING_FILES)

VOCAB_SIZE = len(tokenizer.word_to_id)

# =========================
# Dataset
# =========================


class ConversationDataset(Dataset):
    def __init__(self, jsonl_files, tokenizer):
        self.samples = []

        for jsonl_file in jsonl_files:
            file_samples = 0

            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line_number, line in enumerate(f, start=1):
                    if not line.strip():
                        continue

                    try:
                        data = json.loads(line)
                        text = data["text"]
                    except (json.JSONDecodeError, KeyError) as e:
                        raise ValueError(
                            f"Invalid data in {jsonl_file}, "
                            f"line {line_number}: {e}"
                        ) from e

                    tokens = tokenizer.encode(text)

                    # Skip very short conversations
                    if len(tokens) < 2:
                        continue

                    # Trim to the model's maximum sequence length
                    tokens = tokens[:MAX_LENGTH]

                    self.samples.append(tokens)
                    file_samples += 1

            print(f"{jsonl_file}: {file_samples} conversations")

        print("Total conversations:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens = self.samples[idx]

        x = tokens[:-1]
        y = tokens[1:]

        x = x + [0] * (MAX_LENGTH - 1 - len(x))
        y = y + [0] * (MAX_LENGTH - 1 - len(y))

        return torch.tensor(x), torch.tensor(y)

dataset = ConversationDataset(
    TRAINING_FILES,
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

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model = SimpleGPT(
    vocab_size=VOCAB_SIZE,
    embed_size=EMBED_SIZE,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
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
print("Vocab size:", VOCAB_SIZE)


def save_checkpoint():
    torch.save({
        "model_state": model.state_dict(),
        "vocab": tokenizer.word_to_id,
        "config": {
            "embed_size": EMBED_SIZE,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "max_length": MAX_LENGTH,
        }
    }, MODEL_FILE)


best_loss = float("inf")

for epoch in range(EPOCHS):

    total_loss = 0

    for x, y in loader:

        x = x.to(device)
        y = y.to(device)

        # Mask out padding tokens so they don't act like real content
        padding_mask = (x == 0)

        optimizer.zero_grad()

        outputs = model(x, key_padding_mask=padding_mask)

        loss = criterion(
            outputs.reshape(-1, VOCAB_SIZE),
            y.reshape(-1)
        )

        loss.backward()

        # Prevent exploding gradients, common with small transformers
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)

    print(
        f"Epoch {epoch+1}/{EPOCHS} "
        f"Loss: {avg_loss:.4f}"
    )

    # Keep the best-performing checkpoint, not just whatever the last epoch happened to be
    if avg_loss < best_loss:
        best_loss = avg_loss
        save_checkpoint()

print(f"Model saved to {MODEL_FILE} (best loss: {best_loss:.4f})")
