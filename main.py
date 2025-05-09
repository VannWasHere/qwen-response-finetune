from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
tokenizer.pad_token = tokenizer.eos_token 

# Setup LoRA
lora_config = LoraConfig(
    r=8, lora_alpha=32, lora_dropout=0.1, target_modules=["q_proj", "v_proj"],
    bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

dataset = load_dataset("json", data_files="data/quiz-format.json")

# Define data preprocessing function
def preprocess_function(examples):
    texts = [
        f"### Instruction:\n{instr}\n\n### Response:\n{resp}"
        for instr, resp in zip(examples["instruction"], examples["response"])
    ]
    return tokenizer(texts, padding="max_length", truncation=True, max_length=512)

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./qwen_lora",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=500,
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator
)

trainer.train()
