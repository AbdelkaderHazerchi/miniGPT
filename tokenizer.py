from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

class FastBPETokenizer:
    def __init__(self, vocab_size=5000):
        self.tokenizer = Tokenizer(BPE())
        self.vocab_size = vocab_size
        self.tokenizer.pre_tokenizer = ByteLevel()
        self.token_to_id = {}
        self.id_to_token = {}

    def train(self, files):
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=2,
            special_tokens=[
                "<pad>",
                "<unk>",
                "<|user|>",
                "<|assistant|>",
                "<|end|>",
            ],
        )
        self.tokenizer.train(files, trainer)
        self.token_to_id = self.tokenizer.get_vocab()
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        tokens = []
        for token_id in ids:
            token = self.id_to_token.get(token_id, "")

            if token == "<|end|>":
                break   # ⛔ توقف التوليد هنا

            if token not in ["<pad>", "<unk>"]:
                tokens.append(token)

        text = "".join(tokens)
        text = text.replace("Ġ", " ")
        text = text.replace("Ċ", "\n")
        return text.strip()

    def save(self, path):
        self.tokenizer.save(path)

    def load(self, path):
        self.tokenizer = Tokenizer.from_file(path)
        self.token_to_id = self.tokenizer.get_vocab()
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
