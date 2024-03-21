import spacy
nlp = spacy.load('en_core_web_md')

# Words have vector forms
# nlp(u'lion').vector

doc = nlp('The quick brown fox jumped over the lazy dogs.')
# Docs have vector forms from word vectors
# doc.vector


# Create a three-token Doc object:
tokens = nlp('lion cat pet')

# Iterate through token combinations:
for token1 in tokens:
    for token2 in tokens:
        # Showing word similarity
        # Note: opposites are actually similar, although their meanings are different
        print(token1.text, token2.text, token1.similarity(token2))



tokens = nlp('dog cat nargle')

for token in tokens:
    # Non-defined words will not have a vector, norm of 0, and is out of vocabulary
    print(token.text, token.has_vector, token.vector_norm, token.is_oov)




# Word Vector Addition "king" - "man" + "woman" = "queen"
from scipy import spatial

cosine_similarity = lambda x, y: 1 - spatial.distance.cosine(x, y)

king = nlp.vocab['king'].vector
man = nlp.vocab['man'].vector
woman = nlp.vocab['woman'].vector

# Now we find the closest vector in the vocabulary to the result of "man" - "woman" + "queen"
new_vector = king - man + woman
computed_similarities = []

for word in nlp.vocab:
    # Ignore words without vectors and mixed-case words:
    if word.has_vector:
        if word.is_lower:
            if word.is_alpha:
                similarity = cosine_similarity(new_vector, word.vector)
                computed_similarities.append((word, similarity))

computed_similarities = sorted(computed_similarities, key=lambda item: -item[1])

print([w[0].text for w in computed_similarities[:10]])