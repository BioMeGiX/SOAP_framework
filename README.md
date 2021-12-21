# SOAP_framework
Multi-class Text Annotation and Classification using BERT-based Active Learning
This project has three independent components that works in sequential manner. These three components are represented with three .py files including section_tagger_1.py, active_learning_2.py, and final_model_3.py, respectively.
Description of the files:
1. section_tagger_1.py: code is written for identifying SOAP sections in the text.
2. active_learning_2.py: code is written for transfer active learning that include code for active learning strategies to select instances for labelling and also a BERT based   
   classification model with embedding layer.
3. final_model_3.py: code is written for the final classification model deployed for classifying text instances of SOAP sentences.

Running strategy:
1. Run the section_tagger_1.py on raw data to generate identified SOAP section as an output
2. Run the active_learning_2.py that take initial labeled data (seed data) coming from step 1 and also held-out unlabeled data. This will generate labeled data to use as a training data for the final classificaiotn model.
3. Run the final_model_3.py to create extended embeddings for the final classificaiton model. The code results in trained model that could be used for testing data as we have used for heldout test dataset. The same model could be used for any future application of similar nature.
