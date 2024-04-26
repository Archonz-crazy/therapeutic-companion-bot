from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from torch.utils.data import Dataset
import torch

# Function to load CSV data and return a Dataset object
def load_csv_dataset(file_path, tokenizer, max_length=512):
    data = load_dataset('csv', data_files=file_path)['train']
    return QuestionResponseDataset(tokenizer, data['Question'], data['Response'], max_length)

class QuestionResponseDataset(Dataset):
    def __init__(self, tokenizer, questions, responses, max_length):
        self.tokenizer = tokenizer
        self.questions = questions
        self.responses = responses
        self.max_length = max_length

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        input_text = self.questions[idx]
        target_text = self.responses[idx]
        inputs = self.tokenizer(input_text, max_length=self.max_length, padding="max_length", truncation=True)
        targets = self.tokenizer(target_text, max_length=self.max_length, padding="max_length", truncation=True)
        return {
            'input_ids': torch.tensor(inputs['input_ids']),
            'attention_mask': torch.tensor(inputs['attention_mask']),
            'labels': torch.tensor(targets['input_ids']),
        }

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

# Set padding token if it's not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained("distilgpt2")

file_path = "/home/ubuntu/bot_test/data/sample_q_and_a/train_response.csv"
dataset = load_csv_dataset(file_path, tokenizer)

training_args = TrainingArguments(
    output_dir="./try_results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,  # Reduced batch size
    gradient_accumulation_steps=2,  # Added gradient accumulation
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    logging_dir='./logs',
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
