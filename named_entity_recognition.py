import spacy
nlp = spacy.load('en_core_web_sm')

def show_ents(doc): # Prints all named entities with descriptions
    if doc.ents:
        for ent in doc.ents:
            print(ent.text+' - '+ent.label_+' - '+str(spacy.explain(ent.label_)))
            print(ent.text, ent.start, ent.end, ent.start_char, ent.end_char, ent.label_)
    else:
        print('No named entities found.')

from spacy.tokens import Span
from spacy.matcher import PhraseMatcher


doc = nlp(u'Hello Hector. Hector, may I go to Washington, DC next May to see the thing-thing, thing-thing or thingy-thing?')

# Phrase Matcher
matcher = PhraseMatcher(nlp.vocab)

# Matching Entities
name_phrase_list=['Hector']
product_phrase_list = ['thing-thing', 'thingy-thing']
name_phrase_patterns = [nlp(text) for text in name_phrase_list]
product_phrase_patterns = [nlp(text) for text in product_phrase_list]

# Adding Matches to Matcher
matcher.add('name', None, *name_phrase_patterns)
matcher.add('product', None, *product_phrase_patterns)
matches = matcher(doc)

NAME = doc.vocab.strings[u'PERSON']
PROD = doc.vocab.strings[u'PRODUCT']

# Creating new entities
new_ents = [Span(doc, match[1], match[2], label=NAME) if doc.vocab.strings[match[0]] == 'name' else Span(doc, match[1], match[2], label=PROD) for match in matches]
doc.ents = list(doc.ents) + new_ents
show_ents(doc)

#Counting number of products
print(len([ent for ent in doc.ents if ent.label_=='PRODUCT']))

###############
doc = nlp(u'Originally priced at $29.50, the sweater was @@@@ marked down to five dollars.')
show_ents(doc)
# Quick function to remove ents formed on whitespace:
@spacy.Language.component("remove_special_symbol")
def remove_special_symbol(doc):
    doc.ents = [e for e in doc.ents if not "@" in e.text]
    return doc

# Add the custom function after the NER component
nlp.add_pipe('remove_special_symbol', after='ner')
doc = nlp(u'Originally priced at $29.50, the sweater was @@@@ marked down to five dollars.')
show_ents(doc)
print(nlp.pipe_names)