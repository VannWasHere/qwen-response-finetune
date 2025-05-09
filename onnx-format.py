from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from optimum.onnxruntime import ORTModelForCausalLM
from pathlib import Path

# Define paths
checkpoint_path = "./qwen_lora/checkpoint-102"
base_model_id = "Qwen/Qwen3-1.7B"
output_path = "./qwen_lora_onnx"

print("Loading base model and tokenizer...")
# Load the base model and tokenizer
base_model = AutoModelForCausalLM.from_pretrained(base_model_id)
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

print("Loading LoRA weights...")
# Load the LoRA configuration
peft_config = PeftConfig.from_pretrained(checkpoint_path)

# Apply LoRA weights to the base model
model = PeftModel.from_pretrained(base_model, checkpoint_path)

# Merge LoRA weights with base model for export
print("Merging LoRA weights with base model...")
merged_model = model.merge_and_unload()

# Create output directory if it doesn't exist
Path(output_path).mkdir(exist_ok=True, parents=True)

# First save the merged model in the Transformers format
print("Saving merged model...")
merged_model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

print("Converting to ONNX format...")
# Convert to ONNX using optimum
ort_model = ORTModelForCausalLM.from_pretrained(
    output_path,
    from_transformers=True,
    export=True,
    provider="CPUExecutionProvider"  # Use CPU for export
)

print("Saving ONNX model...")
# Save the ONNX model
ort_model.save_pretrained(output_path)

print(f"Model successfully exported to ONNX format at {output_path}")

# Test the exported model
print("Testing the exported model...")
input_text = "Generate a JSON quiz with 3 questions about"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# Generate with the ONNX model
ort_outputs = ort_model.generate(input_ids, max_length=512)
ort_output_text = tokenizer.decode(ort_outputs[0], skip_special_tokens=True)

print("\nGenerated output:")
print(ort_output_text)
