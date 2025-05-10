from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_model(model_path="VannWasHere/qwen3-tuned-response"):
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    return model, tokenizer

def generate_response(model, tokenizer, instruction):
    input_text = f"<|im_start|>user\nGenerate a JSON quiz based on this instruction: {instruction}<|im_end|>"
    
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
    
    outputs = model.generate(
        input_ids,
        max_length=1024,
        temperature=0.3,
        top_p=0.9,
        repetition_penalty=1.2,
        do_sample=True,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )
    
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "<|im_start|>assistant" in output_text:
        output_text = output_text.split("<|im_start|>assistant")[-1].strip()
    
    return output_text

if __name__ == "__main__":
    # Load the model and tokenizer
    model, tokenizer = load_model()
    
    # Get user input
    instruction = input("Enter your quiz instruction: ")
    
    # Generate and print response
    response = generate_response(model, tokenizer, instruction)
    print("\nModel Response:")
    print(response)



