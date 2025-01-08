# [ ] Fine-tuning

**Status:** To do  
**Priority:** Medium  
**Description:**

Check this one out:

- [Lit-GPT](https://github.com/Lightning-AI/litgpt?tab=readme-ov-file#finetune-an-llm)
- [DataCamp](https://www.datacamp.com/tutorial/fine-tuning-large-language-models)
- [PEFT](https://huggingface.co/docs/peft/en/index)
- [TRL](https://huggingface.co/docs/trl/en/index)

PEFT example:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

# Load your custom dataset from a text file
def load_custom_data(file_path):
    return load_dataset('text', data_files={'train': file_path})

# Specify model and LoRA configuration
model_name = "gpt2"  # Replace with your preferred pre-trained model
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    inference_mode=False,
    r=8,  # Low-rank dimension
    lora_alpha=32,
    lora_dropout=0.1,
)

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Prepare model for k-bit and apply LoRA
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

data_file = "custom_data.txt"  # Path to your text file
raw_datasets = load_custom_data(data_file)
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, remove_columns=["text"])

# Define TrainingArguments
training_args = TrainingArguments(
    output_dir="./lora-finetuned-model",
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=500,
    logging_dir="./logs",
    logging_steps=100,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    fp16=True,  # Enable mixed precision
    push_to_hub=False,
    learning_rate=5e-5,
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["train"],
    tokenizer=tokenizer,
)

# Fine-tune the model
trainer.train()

# Save the model
model.save_pretrained("lora-finetuned-model")
tokenizer.save_pretrained("lora-finetuned-model")
```

TRL example:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset

# Example preference data
preference_data = [
    {"input": "What is AI?", "response_1": "AI is smart.", "response_2": "AI refers to artificial intelligence.", "label": 1},
    {"input": "Tell me a joke.", "response_1": "Why did the chicken cross the road? To get to the other side.", "response_2": "Jokes make people laugh.", "label": 0}
]

# Prepare dataset
def format_for_reward_model(example):
    return {"text": example["response_1"] + " " + example["response_2"], "labels": example["label"]}

dataset = Dataset.from_list(preference_data).map(format_for_reward_model)

# Tokenize
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length")

tokenized_dataset = dataset.map(tokenize, batched=True)

# Load a classification model
reward_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1)

# Train the reward model
training_args = TrainingArguments(
    output_dir="./reward_model",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    evaluation_strategy="epoch"
)

trainer = Trainer(
    model=reward_model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)
trainer.train()


from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOTrainer, PPOConfig

# Load the base language model
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Define PPO configuration
ppo_config = PPOConfig(
    model_name="gpt2",
    learning_rate=1.41e-5,
    batch_size=16,
    forward_batch_size=4
)

# Define a reward function
def reward_fn(query, response):
    # Generate a reward score using the reward model
    inputs = tokenizer(response, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        reward = reward_model(**inputs).logits.squeeze().item()
    return reward

# Initialize PPO Trainer
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    tokenizer=tokenizer,
    reward_model=reward_fn
)

# Prepare example data
example_prompts = [
    "Explain the theory of relativity.",
    "What is the capital of France?",
    "Write a poem about stars."
]

# Tokenize prompts
prompt_tokens = tokenizer(example_prompts, return_tensors="pt", padding=True, truncation=True)

# Perform RL fine-tuning
for epoch in range(3):  # Train for multiple epochs
    for prompt in example_prompts:
        # Generate response
        response = model.generate(prompt_tokens.input_ids)

        # Compute reward
        reward = reward_fn(prompt, tokenizer.decode(response[0]))

        # Perform PPO step
        ppo_trainer.step(prompt, tokenizer.decode(response[0]), reward)
```
