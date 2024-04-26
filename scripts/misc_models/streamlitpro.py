import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage

# Set the API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize the chat model
chat = ChatOpenAI(
    openai_api_key='',
    model='gpt-3.5-turbo'
)

# Use Streamlit to manage the session state
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Streamlit layout
st.title("Therapeutic Companion Bot")

# Ask for the user's name if it's not set
if 'user_name' not in st.session_state:
    user_name = st.text_input("Please enter your name:")
    if user_name:
        st.session_state['user_name'] = user_name
        st.session_state['messages'].append(SystemMessage(content="You are a helpful assistant."))
else:
    # Display the conversation
    def display_messages():
        for message in st.session_state['messages']:
            if isinstance(message, HumanMessage):
                st.write(f"{st.session_state['user_name']}: {message.content}")
            elif isinstance(message, AIMessage):
                st.write(f"Personal AI: {message.content}")

    # Input text box and button in a form for better control of layout and interaction
    with st.form("chat_form"):
        user_input = st.text_input("Talk to the Personal AI:", key="user_input", value="")
        submit_button = st.form_submit_button("Send")

    if submit_button and user_input:  # Check if the button was pressed and there is input
        # Append the human message
        st.session_state['messages'].append(HumanMessage(content=user_input))

        # Get AI response
        response = chat(st.session_state['messages'])

        # Append the AI response
        st.session_state['messages'].append(AIMessage(content=response.content))

    # Display messages after the interaction
    display_messages()
