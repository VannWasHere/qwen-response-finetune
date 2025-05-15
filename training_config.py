from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

def get_training_args(output_dir="./qwen_json_lora"):
    """
    Get training arguments optimized for 4060Ti or similar GPUs
    """
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=6,
        logging_dir="./logs",
        logging_steps=10,
        save_steps=100,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=True,
        save_total_limit=2,
        report_to="none",
        optim="adamw_torch",
        max_grad_norm=1.0,
        weight_decay=0.01,
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )

def setup_trainer(model, tokenizer, train_dataset, val_dataset=None):
    """
    Set up the Trainer with model, tokenizer, and datasets
    """
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Training arguments
    training_args = get_training_args()
    
    # Trainer setup
    trainer_args = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "data_collator": data_collator,
    }
    
    # Add validation dataset if provided
    if val_dataset is not None:
        trainer_args["eval_dataset"] = val_dataset
    
    return Trainer(**trainer_args) 