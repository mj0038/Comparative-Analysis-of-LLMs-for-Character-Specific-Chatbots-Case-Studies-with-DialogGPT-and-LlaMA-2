# evaluate.py
import torch
from datasets import load_metric
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, pipeline
from torch.utils.data import DataLoader
import config
from peft import LoraConfig, PeftModel

def load_model(model_name):
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

  if model_name == config.eval_model:
    model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=config.device_map
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1


    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
  
  else:
    base_model = AutoModelForCausalLM.from_pretrained(
      model_name,
      low_cpu_mem_usage=True,
      return_dict=True,
      torch_dtype=torch.float16,
      device_map=config.device_map,
  )
    model = PeftModel.from_pretrained(base_model, config.new_model)
    model = model.merge_and_unload()

    # Reload tokenizer to save it
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

  return model, tokenizer

# Function to calculate perplexity
def calculate_perplexity(model_name, dataset, device, batch_size=2, max_length=100):
    model, tokenizer = load_model(model_name)
    model.eval()
    total_loss = 0
    total_length = 0

    # DataLoader for batch processing
    loader = DataLoader(dataset, batch_size=batch_size)

    for batch in loader:
        inputs = tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True, max_length=max_length)
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        # Create a mask for non-padding tokens
        non_pad_mask = input_ids != tokenizer.pad_token_id

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss.item()
            loss = loss * non_pad_mask.float()
            loss = loss.sum() / non_pad_mask.sum()

        total_loss += loss * input_ids.size(0)
        total_length += input_ids.size(0)

    average_loss = total_loss / total_length
    perplexity = torch.exp(torch.tensor(average_loss))
    return perplexity.item()

# Function to calculate BLEU score

def clean_and_tokenize(text, tokenizer):
    # Remove special tokens and extra spaces
    text = text.replace("<s>", "").replace("/<s>", "").replace("[INST]", "").replace("[/INST]", "").strip()
    # Tokenize and return the tokens
    return tokenizer.tokenize(text)

def calculate_bleu_score(model_name, dataset, device, batch_size=2, max_length=100):
    model, tokenizer = load_model(model_name)
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=max_length)

    generated_texts = []
    for example in dataset:
        prompt = example["text"]
        result = pipe(prompt)
        cleaned_text = clean_and_tokenize(result[0]['generated_text'], tokenizer)
        generated_texts.append(cleaned_text)

    # Reference texts
    reference_texts = []
    for example in dataset:
        cleaned_ref = clean_and_tokenize(example["text"], tokenizer)
        reference_texts.append([cleaned_ref])  # Note: BLEU expects a list of list for references

    # BLEU metric
    bleu = load_metric("bleu")

    # Calculate BLEU score
    results = bleu.compute(predictions=generated_texts, references=reference_texts)

    return results["bleu"]
