import torch

from tokenizer import SimpleTokenizer
from model import SimpleGPT

# =========================
# Settings
# =========================

MAX_LENGTH = 64
MODEL_FILE = "model.pth"

# =========================
# Load checkpoint
# =========================

checkpoint = torch.load(
    MODEL_FILE,
    map_location=torch.device("cpu")
)

vocab = checkpoint["vocab"]

# =========================
# Rebuild tokenizer
# =========================

tokenizer = SimpleTokenizer()

tokenizer.word_to_id = vocab

tokenizer.id_to_word = {
    idx: word for word, idx in vocab.items()
}

VOCAB_SIZE = len(vocab)

# =========================
# Load model
# =========================

model = SimpleGPT(
    vocab_size=VOCAB_SIZE,
    embed_size=128,
    num_heads=4,
    num_layers=2,
    max_length=MAX_LENGTH
)

model.load_state_dict(
    checkpoint["model_state"]
)

model.eval()

# =========================
# Generate text
# =========================

def generate(prompt, max_new_tokens=30):

    tokens = tokenizer.encode(prompt)

    generated_tokens = []

    for _ in range(max_new_tokens):

        x = torch.tensor(
            [tokens[-MAX_LENGTH:]]
        )

        with torch.no_grad():
            outputs = model(x)

        next_token_logits = outputs[0, -1]

        # Temperature sampling
        temperature = 0.8

        logits = next_token_logits / temperature

        probs = torch.softmax(logits, dim=0)

        next_token = torch.multinomial(
            probs,
            num_samples=1
        ).item()

        tokens.append(next_token)
        generated_tokens.append(next_token)

        word = tokenizer.id_to_word[next_token]

        # Stop at END token
        if word == "<END>":
            break

    # Decode only generated part
    text = tokenizer.decode(generated_tokens)

    # Remove END token text
    text = text.replace("<END>", "")

    # Remove accidental User labels
    if "User:" in text:
        text = text.split("User:")[0]

    # Remove accidental AI label
    text = text.replace("AI:", "")

    return text.strip()

# =========================
# Chat loop
# =========================

print("AI ready.")
print("Type 'quit' to stop.\n")

while True:

    user_input = input("You: ")

    if user_input.lower() == "quit":
        break

    prompt = f"User: {user_input}\nAI:"

    response = generate(prompt)

    print(f"\nAI: {response}")
    print()
