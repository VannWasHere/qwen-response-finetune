import torch
import gc
from model_config import load_model_and_tokenizer
from data_processing import load_data, prepare_datasets
from training_config import setup_trainer

# Clean up memory
gc.collect()
torch.cuda.empty_cache()

def main():
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Print trainable parameters information
    model.print_trainable_parameters()
    
    # Load dataset
    data = load_data("data/quiz-format.json")
    
    # Prepare datasets
    train_dataset, val_dataset = prepare_datasets(data, tokenizer)
    
    # Setup trainer
    trainer = setup_trainer(model, tokenizer, train_dataset, val_dataset)
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()