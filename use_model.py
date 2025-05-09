import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Define paths
checkpoint_path = "./qwen_lora/checkpoint-102"
base_model_id = "Qwen/Qwen3-1.7B"

print("Loading base model and tokenizer...")
# Load the base model and tokenizer
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

print("Loading LoRA weights...")
# Load the LoRA configuration and apply it to the model
model = PeftModel.from_pretrained(base_model, checkpoint_path)

# Optional: Merge LoRA weights with base model for faster inference
print("Merging LoRA weights with base model...")
merged_model = model.merge_and_unload()

# Test the model
print("Testing the fine-tuned model...")
input_text = "Generate a JSON quiz with 3 questions about NextJS"

# Tokenize input
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(merged_model.device)

# Generate output
outputs = merged_model.generate(
    input_ids,
    max_length=1024,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1
)

# Decode and print the output
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\nGenerated output:")
print(output_text)

# Example of how to save the merged model if needed
# merged_model.save_pretrained("./qwen_merged_model")
# tokenizer.save_pretrained("./qwen_merged_model") 