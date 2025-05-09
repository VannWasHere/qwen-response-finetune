from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, TrainerCallback
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import logging
import os
from datetime import datetime

# Configure logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("Starting model training process")
logger.info("Loading Qwen3-1.7B model and tokenizer")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

logger.info("Configuring LoRA parameters")
lora_config = LoraConfig(
    r=8, lora_alpha=32, lora_dropout=0.1, target_modules=["q_proj", "v_proj"],
    bias="none", task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

logger.info("Loading training dataset")
dataset = load_dataset("json", data_files="./data/quiz-format.json")
logger.info(f"Dataset loaded with {len(dataset['train'])} examples")

logger.info("Setting up training arguments")
training_args = TrainingArguments(
    output_dir="./qwen_lora",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=500,
    fp16=True,
    report_to="tensorboard"
)

class LoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            logger.info(f"Training logs: {logs}")
            
    def on_epoch_begin(self, args, state, control, **kwargs):
        logger.info(f"Starting epoch {state.epoch}")
        
    def on_epoch_end(self, args, state, control, **kwargs):
        logger.info(f"Completed epoch {state.epoch}")

logger.info("Initializing trainer")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    callbacks=[LoggingCallback()]
)

logger.info("Starting training")
trainer.train()
logger.info("Training completed")
