# model_training.py
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import config

def load_and_split_dataset():
    # Load and split the dataset
    dataset = load_dataset(config.dataset_name, split="train")

    # Define the split ratio
    train_test_split_ratio = 0.9  # 90% for training, 10% for testing

    # Calculate the number of examples for the training set
    num_train_examples = int(len(dataset) * train_test_split_ratio)

    # Split the dataset
    train_dataset = dataset.select(range(num_train_examples))
    test_dataset = dataset.select(range(num_train_examples, len(dataset)))

    return train_dataset, test_dataset

def train_model():
    train_dataset, test_dataset = load_and_split_dataset()

    # Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=config.use_4bit,
    bnb_4bit_quant_type=config.bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=config.use_nested_quant,
)

# Check GPU compatibility with bfloat16
    if compute_dtype == torch.float16 and config.use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)
    model = AutoModelForCausalLM.from_pretrained(
    config.model_name,
    quantization_config=bnb_config,
    device_map=config.device_map
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        r=config.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Set training parameters
    training_arguments = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        optim=config.optim,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        fp16=config.fp16,
        bf16=config.bf16,
        max_grad_norm=config.max_grad_norm,
        max_steps=config.max_steps,
        warmup_ratio=config.warmup_ratio,
        group_by_length=config.group_by_length,
        lr_scheduler_type=config.lr_scheduler_type,
        report_to="tensorboard"
    )

    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=config.packing,
    )

    # Train model
    trainer.train()

    # Save trained model
    trainer.model.save_pretrained(config.new_model)

    return trainer, tokenizer, test_dataset

# Function to provide access to the test dataset for evaluation
def get_test_dataset():
    train_dataset, test_dataset = load_and_split_dataset()
    return test_dataset
