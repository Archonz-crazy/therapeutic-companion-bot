#%%
# Takes input from the user and classify the question as a knowledge based question or standard questionaire so that the chatbot can access the particular knowledge directory's vectors if the question is not standard or if standard then goes to sample_q_and_a directory's vectors to answer
def classify_question(question):
    # Load the model_d2v_qa.model file
    model_d2v_qa = Doc2Vec.load("../data/sample_q_and_a/preprocess/d2v/model_d2v_qa.model")
    # Load the model_d2v_combined.model file
    model_d2v_combined = Doc2Vec.load("../data/knowledge/preprocess/d2v/model_d2v_combined.model")
    # Tokenize the question
    tokens = question.split()
    # Infer the vector of the question
    vector = model_d2v_qa.infer_vector(tokens)
    # Find the similar vector from the model_d2v_combined.model
    similar_vector = model_d2v_combined.dv.most_similar([vector], topn=1)
    # If the similarity is greater than 0.5 then the question is knowledge based
    if similar_vector[0][1] > 0.5:
        return "knowledge"
    else:
        return "standard"

#%%
#take question from the user
question = input("Enter your question: ")

#%%
#classify the question
classification = classify_question(question)
#display the classification
print(f"The question is {classification} based question")

#%%
#predict the answer from the knowledge or sample_q_and_a corpus and based on classification 
# and predict the answer using RNN and LSTM model
