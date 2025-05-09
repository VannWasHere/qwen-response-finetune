from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

lora_config = LoraConfig(
    r=8, lora_alpha=32, lora_dropout=0.1, target_modules=["q_proj", "v_proj"],
    bias="none", task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# load your JSON quiz prompt-response dataset
dataset = load_dataset("json", data_files="data/quiz-format.json")

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
    train_dataset=dataset
)

trainer.train()
