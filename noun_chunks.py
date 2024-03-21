import spacy
nlp = spacy.load('en_core_web_sm')


doc = nlp(u"Autonomous cars shift insurance liability toward manufacturers.")

for chunk in doc.noun_chunks:
    print(chunk.text+' - '+chunk.root.text+' - '+chunk.root.dep_+' - '+chunk.root.head.text)
print(len(list(doc.noun_chunks)))