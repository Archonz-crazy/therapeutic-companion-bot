import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import random
import json
import pandas as pd
import pickle
import gradio as gr
from tensorflow.python.util.nest import is_sequence_or_composite

stemmer = LancasterStemmer()

with open("/home/ubuntu/bot_test/another/intents.json") as file:
    data = json.load(file)

with open("/home/ubuntu/bot_test/another/data.pickle", "rb") as f:
  words, labels, training, output = pickle.load(f)

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.load("/home/ubuntu/bot_test/another/MentalHealthChatBotmodel.tflearn")
# print('model loaded successfully')


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return np.array(bag)


def chat(message, history):
    history = history or []
    message = message.lower()
    results = model.predict([bag_of_words(message, words)])
    results_index = np.argmax(results)
    tag = labels[results_index]

    for tg in data["intents"]:
      if tg['tag'] == tag:
        responses = tg['responses']

        # print(random.choice(responses))
        response = random.choice(responses)
  
    history.append((message, response))
    return history, history

chatbot = gr.Chatbot(label="Chat")
css = """
footer {display:none !important}
.output-markdown{display:none !important}
.gr-button-primary {
    z-index: 14;
    height: 43px;
    width: 130px;
    left: 0px;
    top: 0px;
    padding: 0px;
    cursor: pointer !important; 
    background: none rgb(17, 20, 45) !important;
    border: none !important;
    text-align: center !important;
    font-family: Poppins !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    color: rgb(255, 255, 255) !important;
    line-height: 1 !important;
    border-radius: 12px !important;
    transition: box-shadow 200ms ease 0s, background 200ms ease 0s !important;
    box-shadow: none !important;
}
.gr-button-primary:hover{
    z-index: 14;
    height: 43px;
    width: 130px;
    left: 0px;
    top: 0px;
    padding: 0px;
    cursor: pointer !important;
    background: none rgb(37, 56, 133) !important;
    border: none !important;
    text-align: center !important;
    font-family: Poppins !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    color: rgb(255, 255, 255) !important;
    line-height: 1 !important;
    border-radius: 12px !important;
    transition: box-shadow 200ms ease 0s, background 200ms ease 0s !important;
    box-shadow: rgb(0 0 0 / 23%) 0px 1px 7px 0px !important;
}
.hover\:bg-orange-50:hover {
    --tw-bg-opacity: 1 !important;
    background-color: rgb(229,225,255) !important;
}
div[data-testid="user"] {
  background-color: #253885 !important;
}
.h-\[40vh\]{
height: 70vh !important;
}
"""
demo = gr.Interface(
    chat,
    [gr.Textbox(lines=1, label="Message"), "state"],
    [chatbot, "state"],
    allow_flagging="never",
    title="Mental Health Bot | Data Science Dojo",
    css=css
)
if __name__ == "__main__":
    demo.launch()