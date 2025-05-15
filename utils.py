import json
import re

EXPECTED_STRUCTURE = {
    "topic": str,
    "questions": [
        {
            "question": str,
            "options": [str, str, str, str],
            "answer": str,
            "hint": str
        }
    ]
}

def validate_json_structure(json_obj):
    """
    Validate that the JSON object has the expected structure.
    Returns a sanitized version of the JSON object or None if invalid.
    """
    if not isinstance(json_obj, dict):
        return None
    
    # Check for required top-level keys
    if "topic" not in json_obj or "questions" not in json_obj:
        return None
    
    if not isinstance(json_obj["topic"], str) or not isinstance(json_obj["questions"], list):
        return None
    
    # Create a sanitized version with the correct structure
    sanitized = {
        "topic": json_obj["topic"],
        "questions": []
    }
    
    # Process each question to ensure correct structure
    for question in json_obj["questions"]:
        if not isinstance(question, dict):
            continue
            
        # Ensure all required fields exist
        if not all(key in question for key in ["question", "options", "answer", "hint"]):
            continue
            
        # Check options array
        if not isinstance(question["options"], list) or len(question["options"]) != 4:
            continue
            
        # Ensure all string fields are strings
        if not all(isinstance(question[key], str) for key in ["question", "answer", "hint"]):
            continue
            
        # Ensure all options are strings
        if not all(isinstance(opt, str) for opt in question["options"]):
            continue
            
        # Add the sanitized question to our result
        sanitized["questions"].append({
            "question": question["question"],
            "options": question["options"],
            "answer": question["answer"],
            "hint": question["hint"]
        })
    
    # If we don't have any valid questions, the JSON is invalid
    if not sanitized["questions"]:
        return None
        
    return sanitized

def normalize_json_response(response):
    """
    Parse and normalize a JSON response to ensure it matches the expected format.
    """
    # First, handle string responses
    if isinstance(response, str):
        # Try to parse as JSON
        try:
            # Remove any leading/trailing markdown or code formatting
            clean_str = re.sub(r'^```(?:json)?|```$', '', response.strip())
            json_obj = json.loads(clean_str)
        except json.JSONDecodeError:
            return None
    else:
        json_obj = response
    
    # Validate and sanitize the structure
    return validate_json_structure(json_obj)

def format_examples(examples):
    """
    Format examples for training, ensuring consistent JSON-only structure
    """
    formatted = []
    
    for ex in examples:
        # Process and normalize the response
        response = ex['response']
        normalized_response = normalize_json_response(response)
        
        if normalized_response is None:
            # Skip invalid examples
            continue
        
        # Serialize the normalized response back to a JSON string with consistent formatting
        # No extra whitespace or indentation - pure JSON
        json_response = json.dumps(normalized_response, ensure_ascii=False, separators=(',', ':'))
        
        # IMPORTANT: Format prompt to teach model to output ONLY JSON with no additional text
        # Use a specific instruction that emphasizes JSON-only output
        prompt = f"<|im_start|>user\nGenerate a quiz in pure JSON format based on this instruction. Respond with JSON only, no additional text: {ex['instruction']}<|im_end|>\n<|im_start|>assistant\n{json_response}<|im_end|>"
        formatted.append({"text": prompt})
    
    return formatted

def tokenize_function(tokenizer, examples):
    """
    Tokenize the examples with proper truncation and padding
    """
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=768,
        padding="max_length",
        return_tensors=None
    ) 