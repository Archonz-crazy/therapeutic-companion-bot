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

