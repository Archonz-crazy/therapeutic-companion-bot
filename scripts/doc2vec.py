#%%
import os
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import multiprocessing
#%%
directory = "../data/knowledge"
d2v_directory = os.path.join(directory, 'd2v')

os.makedirs(d2v_directory, exist_ok=True)
#%%
# Load data
# The path to your single text file
text_file_path = '../data/knowledge/text_preprocess.txt'

# Process the single text file
tagged_data = []

with open(text_file_path, 'r') as file:
    text = file.read()
    #Tokenize the text
    tokens = text.split()
    tagged_data.append(TaggedDocument(words=tokens, tags=["text_file"]))
#%%  
# Initialize and train the Doc2Vec model
model_d2v = Doc2Vec(vector_size=100, alpha=0.025, min_alpha=0.00025, min_count=1, dm=0, workers=multiprocessing.cpu_count(), epochs=100)
model_d2v.build_vocab(tagged_data)

#%%
# Train the Doc2Vec model
model_d2v.train(tagged_data, total_examples=model_d2v.corpus_count, epochs=model_d2v.epochs)

#%%
# Save the model
model_d2v.save(os.path.join(d2v_directory, 'model_d2v.model'))
# %%
#view model_d2v.model file
from gensim.models.doc2vec import Doc2Vec
model = Doc2Vec.load(os.path.join(d2v_directory, 'model_d2v.model'))

#view the document vector
document_tag = "text_file"
document_vector = model.dv[document_tag]
print(f"Vector for document '{document_tag}':\n{document_vector}")

# %%
# View a word vector
word = "example"
word_vector = model.wv[word]
print(f"Vector for word '{word}':\n{word_vector}")
# %%
# List all document tags
all_document_tags = list(model.dv.index_to_key)
print("All document tags:", all_document_tags)

# %%
# Inspect the vocabulary
vocabulary = list(model.wv.key_to_index.keys())
print("Vocabulary:", vocabulary)




# %%
def process_text_file(file_path):
    tagged_data = []
    with open(file_path, 'r') as file:
        text = file.read()
        tokens = text.split()  # Tokenize the text
        tagged_data.append(TaggedDocument(words=tokens, tags=["text_file"]))
    return tagged_data

#%%
def process_csv_files(directory):
    tagged_data = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)
            
            # Identify all columns of type 'object' (string)
            text_columns = [col for col in df.columns if df[col].dtype == 'object']
            
            if not text_columns:
                print(f"No suitable text columns found in {file_path}. Skipping file.")
                continue
            
            print(f"Using columns {text_columns} from {file_path}")
            for index, row in df.iterrows():
                # Concatenate the text from all string columns, separating by a space
                concatenated_text = ' '.join(str(row[col]) for col in text_columns)
                tokens = concatenated_text.split()
                tagged_data.append(TaggedDocument(words=tokens, tags=[f"{os.path.basename(file_path)}_{index}"]))
    return tagged_data

#%%
# Process the text file
from gensim.models.doc2vec import Doc2Vec
import multiprocessing

# Directory paths
directory = "../data/knowledge"
d2v_directory = os.path.join(directory, 'd2v')
os.makedirs(d2v_directory, exist_ok=True)

# Initialize tagged data list
tagged_data = []

# Process the single text file
text_file_path = '../data/knowledge/text_preprocess.txt'
tagged_data.extend(process_text_file(text_file_path))

# Process all CSV files in the directory
tagged_data.extend(process_csv_files(directory))
# Train the Doc2Vec model
model_d2v_combined = Doc2Vec(vector_size=100, alpha=0.025, min_alpha=0.00025, min_count=1, dm=0, workers=multiprocessing.cpu_count(), epochs=100)
model_d2v_combined.build_vocab(tagged_data)
model_d2v_combined.train(tagged_data, total_examples=model_d2v_combined.corpus_count, epochs=model_d2v_combined.epochs)

# Save the model
model_d2v_combined.save(os.path.join(d2v_directory, 'model_d2v_combined.model'))
# %%
