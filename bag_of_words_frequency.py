with open("text1.txt", "w") as file:
    file.write("This is a story about cats\nour feline pets\nCats are furry animals")

with open("text2.txt", "w") as file:
    file.write("This story is about surfing\nCatching waves is fun\nSurfing is a popular water sport")

# Making Dictionary
vocab = {}
i = 1
with open('text1.txt') as f:
    x = f.read().lower().split()
for word in x:
    if word not in vocab:
        vocab[word]=i
        i+=1
with open('text2.txt') as f:
    x = f.read().lower().split()
for word in x:
    if word not in vocab:
        vocab[word]=i
        i+=1
print(vocab)

# Frequency Counter (Further Processing: Term Frequency-Inverse Document Frequency)
one = ['text1.txt']+[0]*len(vocab)
with open('text1.txt') as f:
    x = f.read().lower().split()
for word in x:
    one[vocab[word]]+=1
two= ['text2.txt']+[0]*len(vocab)
with open('text2.txt') as f:
    x = f.read().lower().split()
for word in x:
    two[vocab[word]]+=1
print(one)
print(two)
