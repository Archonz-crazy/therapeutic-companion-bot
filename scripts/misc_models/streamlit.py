from transformers import pipeline

def chatbot():
    # Initialize the pipeline
    pipe = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2")

    print("Hello! I am an AI trained to answer your questions. Ask me anything! (type 'quit' to exit)")
    
    while True:
        # Get input from the user
        question = input("You: ")
        if question.lower() == 'quit':
            break
        
        # Generate a response using the model
        response = pipe(question, max_length=150, num_return_sequences=1)[0]['generated_text']
        
        # Print the generated response
        print("AI:", response)

if __name__ == "__main__":
    chatbot()
