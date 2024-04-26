import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_path):
    """ Load the trained model and tokenizer. """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Set padding token if it's not already set, typically to the eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token is set
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval()  # Put the model in evaluation mode
    return tokenizer, model

def ask_question(question, tokenizer, model):
    """ Generate an answer from the model based on the input question. """
    # Encode the question and convert to tensor with padding and truncation
    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Generate tokens until an end token (tokenizer.eos_token_id) or max_length
    outputs = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    
    # Decode the generated tokens to a string
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

def main():
    # Load the model from the checkpoint
    model_path = "./try_results/checkpoint-4500"  # Specify the correct path
    tokenizer, model = load_model(model_path)

    # Loop to ask questions interactively
    print("Ask me anything! (type 'exit' to quit)")
    while True:
        question = input("Question: ")
        if question.lower() == 'exit':
            break
        answer = ask_question(question, tokenizer, model)
        print("Answer:", answer)

if __name__ == "__main__":
    main()

