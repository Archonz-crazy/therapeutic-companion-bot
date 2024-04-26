import gradio as gr
import os
import spaces
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import torch

# Set an environment variable
HF_TOKEN = os.environ.get("HF_TOKEN", None)

DESCRIPTION = '''
<div>
<h1 style="text-align: center;">Therapeutic Companion Bot</h1>
<p>This is a therapy companion chatbot built using the latest Llama3 model. With this chatbot, you can effectively chat and get answers to questions you have about your mental health and dealing with the daily problems of life.</p>
<p>🔎 For more details about our project, take a look <a href="https://github.com/Archonz-crazy/therapeutic-companion-bot">at our Github repo</a>.</p>
</div>
'''

LICENSE = """
<p/>
---
Built with Meta Llama 3
"""

METRICS = """
<p/>
Perplexity of the model: 4.42

"""

PLACEHOLDER = """
<div style="padding: 30px; text-align: center; display: flex; flex-direction: column; align-items: center;">
   <h1 style="font-size: 28px; margin-bottom: 2px; opacity: 0.55;">Here to help you 😁</h1>
   <p style="font-size: 18px; margin-bottom: 2px; opacity: 0.65;">Ask me anything...</p>
</div>
"""

css = """
h1 {
  text-align: center;
  display: block;
}
#duplicate-button {
  margin: auto;
  color: white;
  background: #060a0f;
  border-radius: 100vh;
}
"""

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", device_map="auto")
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

@spaces.GPU(duration=120)
def chat_llama3_8b(message: str, 
              history: list, 
              temperature: float, 
              max_new_tokens: int
             ) -> str:
    """
    Generate a streaming response using the llama3-8b model.
    Args:
        message (str): The input message.
        history (list): The conversation history used by ChatInterface.
        temperature (float): The temperature for generating the response.
        max_new_tokens (int): The maximum number of new tokens to generate.
    Returns:
        str: The generated response.
    """
    conversation = []
    for user, assistant in history:
        conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
    conversation.append({"role": "user", "content": message})

    input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt").to(model.device)
    
    streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)

    generate_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        eos_token_id=terminators,
        num_return_sequences=1,  # Generate only one response
        top_p=0.9,  # Use nucleus sampling for faster generation
        top_k=50,  # Limit the number of top tokens to consider
    )
    
    if temperature == 0:
        generate_kwargs['do_sample'] = False
        
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    outputs = []
    for text in streamer:
        outputs.append(text)
        yield "".join(outputs)

def calculate_perplexity(model, tokenizer, text):
    encodings = tokenizer(text, return_tensors='pt')

    # Calculate log likelihood of the input text
    with torch.no_grad():
        outputs = model(**encodings, labels=encodings['input_ids'])
        log_likelihood = outputs.loss * encodings['input_ids'].shape[1]

    # Convert to perplexity
    perplexity = torch.exp(log_likelihood)
    return perplexity.item()

text = "This is an example sentence to test the perplexity of the model."

# Calculate perplexity
perplexity = calculate_perplexity(model, tokenizer, text)
print(f"Perplexity of the input text: {perplexity}")

# Gradio block
chatbot = gr.Chatbot(height=450, placeholder=PLACEHOLDER, label='Look here!!')

with gr.Blocks(fill_height=True, css=css) as demo:
    
    gr.Markdown(DESCRIPTION)
    gr.ChatInterface(
        fn=chat_llama3_8b,
        chatbot=chatbot,
        fill_height=True,
        additional_inputs_accordion=gr.Accordion(label="⚙️ Parameters", open=False, render=False),
        additional_inputs=[
            gr.Slider(minimum=0,
                      maximum=1, 
                      step=0.1,
                      value=0.95, 
                      label="Temperature", 
                      render=False),
            gr.Slider(minimum=128, 
                      maximum=4096,
                      step=1,
                      value=512, 
                      label="Max new tokens", 
                      render=False),
            ],
        examples=[
            ['What is depression and how to deal with it?'],
            ['What is life?'],
            ['I have some mental health problems. Please help me.'],
            ['What are the different symptoms of depression'],
            ['What are the major side effects of not dealing with depression']
            ],
        cache_examples=False,
                     )
    gr.Markdown(METRICS)
    
if __name__ == "__main__":
    demo.launch(share=True)