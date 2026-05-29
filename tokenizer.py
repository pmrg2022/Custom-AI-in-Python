import json

class SimpleTokenizer:

    def __init__(self):

        self.word_to_id = {}
        self.id_to_word = {}

    def build_vocab(self, jsonl_file):

        words = set()

        with open(jsonl_file, "r", encoding="utf-8") as f:

            for line in f:

                data = json.loads(line)

                text = data["text"]

                for word in text.split():

                    word = word.strip()

                    if word:
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

        for word in text.split():

            word = word.strip()

            if not word:
                continue

            token = self.word_to_id.get(
                word,
                self.word_to_id["<UNK>"]
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

        return " ".join(words)


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
