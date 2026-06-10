import pandas as pd
from tokenizer import SimpleTokenizer

df = pd.read_csv("../data/emotional/train.csv")

captions = df["caption"].astype(str).tolist()

tokenizer = SimpleTokenizer(vocab_size=10000)

tokenizer.build_vocab(captions)

tokenizer.save("tokenizer_emotional.pkl")

print("Vocabulary Size:", len(tokenizer.word2idx))