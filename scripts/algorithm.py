#%%
import numpy as np
import gensim
from gensim.models.doc2vec import Doc2Vec
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
#%%
def is_knowledge_based(question, knowledge_model, standard_model):
    """
    Determines if a question is knowledge-based or standard by comparing similarity scores
    from two Doc2Vec models.
    """
    question_vector_knowledge = knowledge_model.infer_vector(question.split())
    question_vector_standard = standard_model.infer_vector(question.split())

    similarity_knowledge = abs(question_vector_knowledge).sum()
    similarity_standard = abs(question_vector_standard).sum()
    print("knowledge similarity", similarity_knowledge, "standard similarity", similarity_standard)
    return similarity_knowledge > similarity_standard
#%%
def classify_question(question, knowledge_model, standard_model):
    if is_knowledge_based(question, knowledge_model, standard_model):
        return "knowledge-based"
    else:
        return "standard"
#%%
# Define the directories for the models
knowledge_model_dir = "../data/knowledge/preprocess/d2v/model_d2v_combined.model"
standard_model_dir = "../data/sample_q_and_a/preprocess/d2v/model_d2v_qa.model"

# Load the Doc2Vec models
knowledge_model = Doc2Vec.load(knowledge_model_dir)
standard_model = Doc2Vec.load(standard_model_dir)
#%%
# Take question from the user
question = input("Enter your question: ")

# Classify the question
classification = classify_question(question, knowledge_model, standard_model)

# Display the classification
print(f"The question is a {classification} question.")

#%%
#predict the answer from the knowledge or sample_q_and_a corpus and based on classification 
# and predict the answer using LSTM model
knowledge_model_dir = "../data/knowledge/preprocess/d2v/model_d2v_combined.model"
standard_model_dir = "../data/sample_q_and_a/preprocess/d2v/model_d2v_qa.model"
# Load the Doc2Vec models
knowledge_model = gensim.models.doc2vec.Doc2Vec.load(knowledge_model_dir)
standard_model = gensim.models.doc2vec.Doc2Vec.load(standard_model_dir)

knowledge_vectors = knowledge_model.infer_vector(question.split())
standard_vectors = standard_model.infer_vector(question.split())

combined_vectors = np.hstack((knowledge_vectors, standard_vectors))

#%%
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
from keras.optimizers import Adam
#%%
# Define model parameters
vocab_size = len(combined_vectors.wv.vocab)
seq_length = 5000  # Length of input sequences
embedding_dim = 300  # Size of the embedding vector
sequence_length = 50  # Number of documents in each input sequence

#%%
X_train = []
Y_train = []

for i in range(len(combined_vectors) - sequence_length):
    sequence = combined_vectors[i:i+sequence_length]
    next_vector = combined_vectors[i + sequence_length]
    
    X_train.append(sequence)
    Y_train.append(next_vector)

X_train = np.array(X_train)
Y_train = np.array(Y_train)

#%%
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=seq_length))
model.add(LSTM(128, return_sequences=False))
model.add(Dense(vocab_size, activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=128, epochs=10, validation_split=0.1)

#%%
def generate_text(seed_text, model, tokenizer, seq_length, num_generated=100):
    result = list(seed_text)
    for _ in range(num_generated):
        encoded = tokenizer.texts_to_sequences([seed_text])[0]
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        
        # Predict probabilities for each token
        probabilities = model.predict(encoded, verbose=0)[0]
        predicted_token = np.argmax(probabilities)
        
        # Convert token to word and add to the result
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == predicted_token:
                out_word = word
                break
        seed_text += ' ' + out_word
        result.append(out_word)
    return ' '.join(result)
