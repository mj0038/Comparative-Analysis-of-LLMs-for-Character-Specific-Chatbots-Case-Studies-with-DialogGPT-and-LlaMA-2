import torch
from transformers import AutoModelWithLMHead, AutoTokenizer, get_linear_schedule_with_warmup

def chat_with_bot(model, tokenizer, chat_history_ids, num_lines=5):
    for step in range(num_lines):
        new_user_input_ids = tokenizer.encode(input(">> User:") + tokenizer.eos_token, return_tensors='pt')

        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

        chat_history_ids = model.generate(
            bot_input_ids, max_length=200,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=100,
            top_p=0.7,
            temperature=0.8
        )

        print("RickBot: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))

tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')
model = AutoModelWithLMHead.from_pretrained('output-small')
chat_history_ids = torch.tensor([])  # Initialize chat history

chat_with_bot(model, tokenizer, chat_history_ids, num_lines=5)
