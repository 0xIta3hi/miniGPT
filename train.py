with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print("the length of the text is:", len(text))

# total unique characters that occur in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
print(encode("hii there"))
print(decode(encode("hii there")))

import torch
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000])

# train test split
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]
