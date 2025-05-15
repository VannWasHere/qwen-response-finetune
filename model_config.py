from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

def get_bnb_config():
    """
    Get BitsAndBytes configuration for 4-bit quantization
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

def get_lora_config():
    """
    Get memory-efficient LoRA configuration for fine-tuning
    """
    return LoraConfig(
        r=8,
        lora_alpha=16, 
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode=False,
    )

def load_model_and_tokenizer(model_name="Qwen/Qwen3-1.7B"):
    """
    Load model and tokenizer with quantization and prepare for training
    """
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=get_bnb_config(),
        device_map="auto"
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare model for training
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    
    # Apply LoRA
    model = get_peft_model(model, get_lora_config())
    
    return model, tokenizer 