import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re

# Define paths
checkpoint_path = "./qwen_json_lora/checkpoint-85"  # Use the new model path
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

# Merge LoRA weights with base model for faster inference
print("Merging LoRA weights with base model...")
merged_model = model.merge_and_unload()

def generate_json_quiz(instruction, max_retries=3):
    """
    Generate a JSON quiz using the fine-tuned model.
    Will retry if output is not valid JSON.
    """
    # Format the input with specific instruction to get JSON only
    input_text = f"<|im_start|>user\nGenerate a JSON quiz based on this instruction: {instruction}<|im_end|>"
    
    # Tokenize input
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(merged_model.device)
    
    for attempt in range(max_retries):
        # Generate output with strict parameters to ensure cleaner generation
        outputs = merged_model.generate(
            input_ids,
            max_length=1024,
            temperature=0.3,  # Lower temperature for more deterministic output
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Decode the output
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the JSON part - look for content after "assistant" marker
        if "<|im_start|>assistant" in output_text:
            output_text = output_text.split("<|im_start|>assistant")[-1].strip()
        
        # Try to extract JSON using regex
        json_pattern = r'(\{.*\})'
        json_matches = re.findall(json_pattern, output_text, re.DOTALL)
        
        if json_matches:
            # Take the largest JSON match (most complete)
            json_text = max(json_matches, key=len)
            
            # Verify it's valid JSON
            try:
                json_obj = json.loads(json_text)
                return json_obj
            except json.JSONDecodeError:
                print(f"Attempt {attempt+1}: Generated invalid JSON. Retrying...")
                continue
        
        print(f"Attempt {attempt+1}: No valid JSON found. Retrying...")
    
    print("Failed to generate valid JSON after multiple attempts.")
    return {"error": "Failed to generate valid JSON"}

# Test the model with different topics
test_topics = [
    "Generate a 3-question multiple choice quiz about Artificial Intelligence",
    "Generate a JSON quiz with 4 questions about Web Development",
    "Generate a 5-question multiple choice quiz about Data Science in JSON format"
]

for topic in test_topics:
    print(f"\n\nTesting: {topic}")
    result = generate_json_quiz(topic)
    print(json.dumps(result, indent=2)) 