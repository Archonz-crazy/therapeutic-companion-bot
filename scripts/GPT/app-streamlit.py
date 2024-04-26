#%%
import streamlit as st
import pandas as pd

#%%
def load_model(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return model


def load_tokenizer(tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    return tokenizer

def trim_incomplete_sentence(text):
    """
    Trims incomplete sentence from the end of the generated text.
    """
    # Attempt to find the last complete sentence
    for punct in [",",".","?","!", ";", ": ", ".\n", "?\n", "!\n", ";\n", ":\n", "\n", "\n\n", "\n\n\n", ", ", ". ", "? ", "! ", "; ", ": "]:
        last_punct_pos = text.rfind(punct)
        if last_punct_pos != -1:
            return text[:last_punct_pos + 1]
    return text  # Return the original text if no sentence-ending punctuation is found

def trim_extra_text(generated_text, max_sentences=2):
    sentences = sent_tokenize(generated_text)
    trimmed_text = ' '.join(sentences[:max_sentences])
    #remove sentence after inverted commas
    if '"' in trimmed_text:
        trimmed_text = trimmed_text[:trimmed_text.rfind('"')+1]
    return trimmed_text

def generate_text(model_path, sequence, max_length, temperature=0.7):
    
    model = load_model(model_path)
    tokenizer = load_tokenizer(model_path)
    ids = tokenizer.encode(f'{sequence}', return_tensors='pt')
    final_outputs = model.generate(
        ids,
        do_sample=True,
        temperature = temperature,
        max_length=max_length,
        pad_token_id=model.config.eos_token_id,
        eos_token_id=model.config.eos_token_id,
        top_k=90,
        top_p=0.6,
        early_stopping=True,  # Stops generation when EOS token is produced
        no_repeat_ngram_size=2
    )
    # Decoding the generated text
    generated_text = tokenizer.decode(final_outputs[0], skip_special_tokens=True)
    
    #removing question part from generated text
    clean_generated_text = generated_text[len(tokenizer.decode(ids[0], skip_special_tokens=True)):].strip()
    
    # Post-generation cleanup
    clean_generated_text = clean_generated_text.replace('","', '')
    clean_generated_text = clean_generated_text.replace(',', '')# Removing specific unwanted characters
    clean_generated_text = trim_incomplete_sentence(clean_generated_text)  # Trimming incomplete sentences
    clean_generated_text = trim_extra_text(clean_generated_text, max_sentences=3)  # Trimming extra text


    return clean_generated_text

#%%
data = pd.read_csv("../data/sample_q_and_a/train_response.csv")
model_path = "../data/sample_q_and_a/gpt-model"
sequence = data['Question'].iloc[660]
max_len = 100
print("Q : ",data['Question'].iloc[660])
print()
print("A : ",data['Response'].iloc[660])
print()
print("G : ",generate_text(model_path, sequence, max_len))

#%%
# Define keywords for detecting crisis situations
CRISIS_KEYWORDS = ['suicide', 'selfharm', 'self-harm', 'assault', 'suicidal']

# Define a function to check for crisis keywords and return appropriate responses
def check_for_crisis(text):
    if any(keyword in text.lower() for keyword in CRISIS_KEYWORDS):
        return ("If you're in crisis or think you may have an emergency, call your doctor or 911 immediately. "
                "If you're having suicidal thoughts, call 1-800-273-TALK (1-800-273-8255) in the U.S. "
                "to talk to a skilled, trained counselor at a crisis center in your area at any time (National Suicide Prevention Lifeline). "
                "If you are located outside the United States, call your local emergency line immediately.")
    return None

st.title('Mental Health Support Chat')
#%%
# Greeting message at the start of the chat
if 'start_message' not in st.session_state:
    st.session_state['start_message'] = "Hi! How are you feeling today?"

# Goodbye message at the end of the chat
if 'end_message' not in st.session_state:
    st.session_state['end_message'] = "It was nice talking to you. Have a good day!"

# Display the start message once at the beginning
if 'start_message' in st.session_state and st.session_state['start_message']:
    st.write(st.session_state['start_message'])
    st.session_state['start_message'] = None  # Clear the message so it doesn't show again

# Text input for user's message
user_input = st.text_input("Enter your message or type 'bye' to end the chat:")

# Main chat function
if st.button('Send'):
    # End the chat if the user says goodbye
    if user_input.lower() in ["bye", "goodbye", "quit"]:
        st.write(st.session_state['end_message'])
        st.stop()

    # Check for crisis situations and respond accordingly
    crisis_response = check_for_crisis(user_input)
    if crisis_response:
        st.write(crisis_response)
        st.stop()

    # Otherwise, generate a response using the model
    else:
        # Set the model path (adjust as necessary)
        model_path = "../data/sample_q_and_a/gpt-model"
        generated_response = generate_text(model_path, user_input, max_len=100)
        st.text_area("AI Response:", value=generated_response, height=150)
