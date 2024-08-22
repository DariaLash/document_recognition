import spacy
from spacy.tokens import DocBin
from spacy.training.example import Example
from spacy.util import minibatch, compounding

# training data
TRAINING_DATA = [
    ("Quote Line No.", {"entities": [(0, 14, "Position")]}),
    ("Qrd Qty", {"entities": [(0, 7, "Quantity")]}),
    ("Qrd Uom", {"entities": [(0, 7, "UOM")]}),
    ("Samsung S20 6/128 Pink", {"entities": [(0, 7, "Vendor"), (8, 11, "Model")]}),
    ("Samsung S22 8/256 Red", {"entities": [(0, 7, "Vendor"), (8, 11, "Model")]}),
]

#to convert training data to spaCy format
def convert_to_spacy_format(training_data):
    nlp = spacy.blank("en")
    db = DocBin()
    for text, annotations in training_data:
        doc = nlp.make_doc(text)
        entities = annotations.get("entities")
        spans = [doc.char_span(start, end, label=label) for start, end, label in entities]
        valid_spans = [span for span in spans if span is not None]
        doc.ents = valid_spans
        db.add(doc)
    return db

db = convert_to_spacy_format(TRAINING_DATA)
db.to_disk("./training_data.spacy")
# Load a blank model
nlp = spacy.blank("en")

#a new NER component
ner = nlp.add_pipe('ner')

#add the labels to the 'ner' component
for _, annotations in TRAINING_DATA:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

#the training data
train_data = [(text, annotations) for text, annotations in TRAINING_DATA]

#start the training
optimizer = nlp.begin_training()
for itn in range(50):  # Number of iterations
    losses = {}
    batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
    for batch in batches:
        texts, annotations = zip(*batch)
        examples = [Example.from_dict(nlp.make_doc(text), annot) for text, annot in zip(texts, annotations)]
        nlp.update(examples, sgd=optimizer, drop=0.5, losses=losses)
    print(f"Losses at iteration {itn}: {losses}")

#save the trained model
nlp.to_disk("./model")

#test the trained model
test_text = "Samsung S22 8/256 Green"
doc = nlp(test_text)
print(f"Entities in '{test_text}':")
for ent in doc.ents:
    print(f"{ent.label_}: {ent.text}")
