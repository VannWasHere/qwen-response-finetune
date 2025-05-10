from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import torch
import json
import gc

# Free up memory
gc.collect()
torch.cuda.empty_cache()

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-1.7B",
    quantization_config=bnb_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
tokenizer.pad_token = tokenizer.eos_token

model.config.use_cache = False
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16, 
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

# Process and normalize the JSON responses with hints
def normalize_json_response(response):
    """
    Ensure the response is a properly formatted JSON object with hints in each question
    """
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except json.JSONDecodeError:
            cleaned = response.strip('`').strip()
            if cleaned.startswith('json'):
                cleaned = cleaned[4:].strip()
            try:
                response = json.loads(cleaned)
            except json.JSONDecodeError:
                return None

    # Ensure each question has a hint
    if isinstance(response, dict) and "questions" in response:
        for question in response["questions"]:
            if "hint" not in question or not question["hint"]:
                # Add a default hint if missing
                if "answer" in question:
                    answer = question["answer"]
                    question["hint"] = f"Hint related to the answer: {answer}"
                else:
                    question["hint"] = "Think carefully about this question"
    
    return response

# Define data preprocessing function with standardized JSON format including hints
def format_examples(examples):
    formatted = []
    
    for ex in examples:
        # Process and normalize the response
        response = ex['response']
        normalized_response = normalize_json_response(response)
        
        if normalized_response is None:
            # Skip invalid examples
            continue
        
        # Serialize the normalized response back to a JSON string with consistent formatting
        json_response = json.dumps(normalized_response, ensure_ascii=False)
        
        # Format to emphasize direct JSON output without any text
        prompt = f"<|im_start|>user\nGenerate a JSON quiz with hints based on this instruction: {ex['instruction']}<|im_end|>\n<|im_start|>assistant\n{json_response}<|im_end|>"
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
        max_length=768,
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

# Training arguments - optimized for 4060Ti
training_args = TrainingArguments(
    output_dir="./qwen_json_lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=6,
    logging_steps=5,
    save_steps=100,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=True,
    save_total_limit=2,
    report_to="none",
    optim="adamw_torch",
    max_grad_norm=1.0,
    weight_decay=0.01,
    remove_unused_columns=False,
    gradient_checkpointing=True,
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

# Save final model
trainer.save_model("./qwen_json_lora_final")