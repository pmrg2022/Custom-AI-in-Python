import json
import re

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

    def build_vocab(self, jsonl_file):

        words = set()

        with open(jsonl_file, "r", encoding="utf-8") as f:

            for line in f:

                data = json.loads(line)

                text = data["text"]

                for word in self._split(text):
                    words.add(word)

        # Special tokens
        # Remove special tokens if already present
        words.discard("<PAD>")
        words.discard("<UNK>")
        words.discard("<END>")

        vocab = [
            "<PAD>",
            "<UNK>",
            "<END>"
        ] + sorted(list(words))

        # Create vocab dictionaries
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

    tokenizer = SimpleTokenizer()

    tokenizer.build_vocab(
        "training_data.jsonl"
    )

    sample = "User: Hello"

    encoded = tokenizer.encode(sample)

    print("Encoded:", encoded)

    decoded = tokenizer.decode(encoded)

    print("Decoded:", decoded)
