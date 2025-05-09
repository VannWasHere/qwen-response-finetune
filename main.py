from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, TrainerCallback
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import logging
import os
from datetime import datetime
import json

# Configure logging
def setup_logging(log_dir="logs"):
    """Set up logging configuration"""
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
    return logging.getLogger(__name__)

def validate_dataset(dataset_path):
    """Validate if dataset exists and has correct format"""
    logger = logging.getLogger(__name__)
    
    # Check if dataset file exists
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset file not found: {dataset_path}")
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    # Validate dataset file
    try:
        with open(dataset_path, 'r') as f:
            data = json.load(f)
            if not data or not isinstance(data, dict) or 'train' not in data:
                logger.error(f"Dataset file is invalid or empty: {dataset_path}")
                raise ValueError(f"Dataset file is invalid or empty: {dataset_path}")
            logger.info(f"Dataset validation passed: {len(data.get('train', []))} training examples found")
    except json.JSONDecodeError:
        logger.error(f"Dataset file contains invalid JSON: {dataset_path}")
        raise ValueError(f"Dataset file contains invalid JSON: {dataset_path}")
    
    return True

def load_model_and_tokenizer(model_name):
    """Load model and tokenizer"""
    logger = logging.getLogger(__name__)
    logger.info(f"Loading {model_name} model and tokenizer")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info(f"Successfully loaded {model_name}")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load model and tokenizer: {str(e)}")
        raise

def prepare_lora_model(model, lora_config_args):
    """Prepare LoRA model with given configuration"""
    logger = logging.getLogger(__name__)
    logger.info("Configuring LoRA parameters")
    
    lora_config = LoraConfig(**lora_config_args)
    model = get_peft_model(model, lora_config)
    logger.info("LoRA model configured successfully")
    return model

def load_training_data(dataset_path):
    """Load and prepare training dataset"""
    logger = logging.getLogger(__name__)
    logger.info("Loading training dataset")
    
    try:
        dataset = load_dataset("json", data_files=dataset_path)
        logger.info(f"Dataset loaded with {len(dataset['train'])} examples")
        return dataset
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        raise

class LoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            logging.getLogger(__name__).info(f"Training logs: {logs}")
            
    def on_epoch_begin(self, args, state, control, **kwargs):
        logging.getLogger(__name__).info(f"Starting epoch {state.epoch}")
        
    def on_epoch_end(self, args, state, control, **kwargs):
        logging.getLogger(__name__).info(f"Completed epoch {state.epoch}")

def train_model(model, tokenizer, dataset, training_args_dict, output_dir="./model_output"):
    """Train the model with the given dataset and arguments"""
    logger = logging.getLogger(__name__)
    logger.info("Setting up training arguments")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        **training_args_dict
    )
    
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
    return trainer

def main():
    # Setup logging
    logger = setup_logging()
    logger.info("Starting model training process")
    
    # Configuration
    model_name = "Qwen/Qwen3-1.7B"
    dataset_path = "./data/quiz-format.json"
    output_dir = "./qwen_lora"
    
    lora_config_args = {
        "r": 8, 
        "lora_alpha": 32, 
        "lora_dropout": 0.1, 
        "target_modules": ["q_proj", "v_proj"],
        "bias": "none", 
        "task_type": "CAUSAL_LM"
    }
    
    training_args_dict = {
        "per_device_train_batch_size": 2,
        "num_train_epochs": 3,
        "logging_steps": 10,
        "save_steps": 500,
        "fp16": True,
        "report_to": "tensorboard"
    }
    
    validate_dataset(dataset_path)
    
    model, tokenizer = load_model_and_tokenizer(model_name)
    
    model = prepare_lora_model(model, lora_config_args)
    
    dataset = load_training_data(dataset_path)
    
    trainer = train_model(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        training_args_dict=training_args_dict,
        output_dir=output_dir
    )
    
    return trainer

if __name__ == "__main__":
    main()
