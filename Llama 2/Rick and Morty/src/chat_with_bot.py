# chat_with_bot.py
import torch
from transformers import pipeline, AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
import config
from peft import LoraConfig, PeftModel

# Function to chat with the bot

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

def chat_with_bot(model_name):
    model, tokenizer = load_model(model_name)
    text_generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
    print("Starting chat with pre-existing fine-tuned bot. Type 'quit' to exit.")
    conversation_history = ""

    while True:
        user_input = input(">> User: ")
        if user_input.lower() == 'quit':
            break

        model_input = conversation_history + f"<s>[INST] {user_input} [/INST]"
        response = text_generator(model_input)
        bot_response = response[0]['generated_text'].strip()
        print("Model Bot: " + bot_response)
