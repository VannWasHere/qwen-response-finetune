from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch  # Ensure to import torch for GPU handling

# Step 1: Load the Base Model and Tokenizer
model_name = "Qwen/Qwen3-1.7B"  # Change this if you have a different base model
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Step 2: Load the LoRA Configuration and Apply It
# Load the fine-tuned LoRA model from the checkpoint
peft_config = PeftConfig.from_pretrained("./qwen_lora/checkpoint-102")
model = PeftModel.from_pretrained(model, "./qwen_lora/checkpoint-102")

# Step 3: Set Device (GPU or CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available, otherwise use CPU
model = model.to(device)  # Move the model to the selected device

# Step 4: Use the Model for Inference
input_text = "Generate a JSON quiz with 5 questions about Artificial Intelligence."
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)  # Move input data to the device

# Generate output with the model
outputs = model.generate(input_ids, max_length=1024)

# Step 5: Decode and Display the Output
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
