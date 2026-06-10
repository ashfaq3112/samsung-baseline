from tokenizer import SimpleTokenizer

tokenizer = SimpleTokenizer()
tokenizer.load("tokenizer_emotional.pkl")

test = "He scales the icy cliff with fearless determination"

tokens = tokenizer.encode(test)

print(tokens)
print(tokenizer.decode(tokens))