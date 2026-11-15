import json
import re
from pathlib import Path

# Special tokens are matched whole, first, so they stay single tokens
# instead of being torn apart into "<", "END", ">" by the general rules
# below. Then: words (with optional contractions like "don't"), or
# single punctuation characters, so punctuation no longer gets glued
# onto words as separate vocab entries (e.g. "Hello" vs "Hello.").
TOKEN_PATTERN = re.compile(r"<PAD>|<UNK>|<END>|\w+(?:'\w+)?|[^\w\s]")


class SimpleTokenizer:

    def __init__(self):

        self.word_to_id = {}
        self.id_to_word = {}

    def _split(self, text):
        return TOKEN_PATTERN.findall(text)

    
    def build_vocab(self, jsonl_files):
        # Allow either one file or a list of files
        if isinstance(jsonl_files, (str, Path)):
            jsonl_files = [jsonl_files]

        words = set()

        for jsonl_file in jsonl_files:
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line_number, line in enumerate(f, start=1):
                    if not line.strip():
                        continue

                    try:
                        data = json.loads(line)
                        text = data["text"]
                    except (json.JSONDecodeError, KeyError) as e:
                        raise ValueError(
                            f"Invalid training data in {jsonl_file}, "
                            f"line {line_number}: {e}"
                        ) from e

                    for word in self._split(text):
                        words.add(word)

        # Keep special tokens at their existing IDs
        words.discard("<PAD>")
        words.discard("<UNK>")
        words.discard("<END>")

        vocab = [
            "<PAD>",
            "<UNK>",
            "<END>"
        ] + sorted(words)

        self.word_to_id = {}
        self.id_to_word = {}

        for idx, word in enumerate(vocab):
            self.word_to_id[word] = idx
            self.id_to_word[idx] = word

        print("Vocabulary size:", len(self.word_to_id))
        print("Highest token ID:", max(self.word_to_id.values()))

    def encode(self, text):

        tokens = []

        for word in self._split(text):

            token = self.word_to_id.get(
                word,
                self.word_to_id.get("<UNK>", 1)
            )

            tokens.append(token)

        return tokens

    def decode(self, tokens):

        words = []

        for token in tokens:

            word = self.id_to_word.get(
                token,
                "<UNK>"
            )

            words.append(word)

        text = " ".join(words)

        # Tidy up spacing so punctuation hugs the word before it
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)

        return text


# =========================
# Test
# =========================

if __name__ == "__main__":

    from pathlib import Path

    training_dir = Path("training_data")
    training_files = sorted(training_dir.glob("*.jsonl"))

    if not training_files:
        print(f"No .jsonl files found in: {training_dir.resolve()}")
    else:
        tokenizer = SimpleTokenizer()
        tokenizer.build_vocab(training_files)

    sample = "User: Hello"

    encoded = tokenizer.encode(sample)

    print("Encoded:", encoded)

    decoded = tokenizer.decode(encoded)

    print("Decoded:", decoded)

