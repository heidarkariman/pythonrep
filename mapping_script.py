import spacy
from spacy import displacy

# Load spaCy's German model
nlp = spacy.load("de_core_news_lg")

# Load input text and output text files
with open("input.txt", "r") as f:
    input_text = f.read()

with open("output.txt", "r") as f:
    output_text = f.read()

# Process input text with spaCy
doc = nlp(input_text)

# Extract relevant information from input text
categories = []
values = []
for ent in doc.ents:
    categories.append(ent.label_)
    values.append(ent.text)

# Process output text with spaCy
doc = nlp(output_text)

# Extract relevant information from output text
entities = {}
for ent in doc.ents:
    entities[ent.text] = ent.label_

# Map categories to entities
for entity, label in entities.items():
    if label in categories:
        idx = categories.index(label)
        entities[entity] = values[idx]

# Print final mapping
for entity, label in entities.items():
    print(f"{entity}: {label}")
