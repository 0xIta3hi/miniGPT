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
# print(data[:1000])

# train test split
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# difining block_size or context window
block_size = 8
print(train_data[:block_size+1])

x = train_data[:block_size] # block size of the characters
y = train_data[1:block_size+1] # next block size of the characters
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"When input is {context} the target: {target}")

torch.manual_seed(1337)
batch_size = 4
block_size = 8

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x,y

xb,yb = get_batch('train')
print("inputs:")
print(xb.shape)
print('targets : ')
print(yb.shape)
print(yb)

print('----')

for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        target = yb[b,t]
        print(f"When input is {context.tolist()} the target: {target}")

    