from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import torch
import json
import gc

# Free up memory
gc.collect()
torch.cuda.empty_cache()

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-1.7B",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
tokenizer.pad_token = tokenizer.eos_token

# Prepare model for training
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)

# Use stronger LoRA configuration targeting more parameters
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    bias="none",
    task_type="CAUSAL_LM",
    inference_mode=False,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Load dataset
with open("data/quiz-format.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Define data preprocessing function - focus on direct instruction-to-JSON mapping
def format_examples(examples):
    formatted = []
    
    for ex in examples:
        # Format to emphasize direct JSON output without reasoning
        prompt = f"<|im_start|>user\nGenerate a JSON quiz based on this instruction: {ex['instruction']}<|im_end|>\n<|im_start|>assistant\n{ex['response']}<|im_end|>"
        formatted.append({"text": prompt})
    
    return formatted

# Format the examples
formatted_examples = format_examples(data)

# Create dataset
from datasets import Dataset
train_dataset = Dataset.from_list(formatted_examples)

# Tokenize function
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=1024,
        padding="max_length",
        return_tensors=None
    )

# Tokenize the dataset
tokenized_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Training arguments - longer training with more steps
training_args = TrainingArguments(
    output_dir="./qwen_json_lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=5,  # Increased epochs
    logging_steps=5,
    save_steps=50,  # Save more frequently
    learning_rate=1e-5,  # Lower learning rate for better convergence
    warmup_steps=100,
    fp16=True,
    save_total_limit=3,
    report_to="none",
    optim="adamw_torch",
    max_grad_norm=1.0,
    weight_decay=0.01,
    remove_unused_columns=False,  # Important for custom datasets
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Start training
trainer.train()
