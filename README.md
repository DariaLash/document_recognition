# document_recognition
One of my experiment how to recognize info from tables in picture 

The main task - to recognize text in tables from the picture, keep where the text located in the page and create the final json with items from and positions, put them to the exel table

The text was already recognized by tesseract model

The final json (what located where) in the final.json
The main file in main.py
The train_model.py for model training.

I created a model based on small dataset of entities  that the main target to use this model: to split vendor and model. I also added some headers to the entities to check how it works. But the Price header not included to the model - because we also have Unit price in the other rows. 
