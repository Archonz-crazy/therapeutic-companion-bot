#%%
import numpy as np
import gensim
from gensim.models.doc2vec import Doc2Vec
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import os
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
# Convert the question into a sequence of word embeddings
question_sequence = knowledge_model.infer_vector(question.split())

# Reshape the question sequence to match the input shape of the LSTM model
question_sequence = np.reshape(question_sequence, (1, question_sequence.shape[0], 1))

# Create a folder called "model" in the "data" directory
model_dir = "../data/model"
os.makedirs(model_dir, exist_ok=True)

# Load the LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(200, input_shape=(question_sequence.shape[1], 1)))
lstm_model.add(Dense(600, activation='sigmoid'))
lstm_model.add(Dense(500, activation='sigmoid'))
lstm_model.add(Dense(400, activation='sigmoid'))
lstm_model.add(Dense(300, activation='sigmoid'))
lstm_model.add(Dense(200, activation='sigmoid'))
lstm_model.add(Dense(100, activation='sigmoid'))
lstm_model.add(Dense(50, activation='sigmoid'))
lstm_model.add(Dense(1, activation='sigmoid'))

lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Save the LSTM model weights to the "model.h5" file in the "model" directory
lstm_model.save(os.path.join(model_dir, "lstm_model.h5"))

# Load the weights of the LSTM model from the standard_model
lstm_model.load_weights(os.path.join(model_dir, "lstm_model.h5"))

#fit the model
lstm_model.fit(question_sequence, np.array([0]), epochs=20, batch_size=1)
# Generate the answer using the LSTM model
answer = lstm_model.predict(question_sequence)

# Display the answer
print(f"The answer is: {answer}")
# %%
#decode the answer to text
answer = answer.reshape(-1)
answer_text = knowledge_model.wv.most_similar(positive=[answer], topn=1)
print(f"The answer is: {answer_text}")
# %%
