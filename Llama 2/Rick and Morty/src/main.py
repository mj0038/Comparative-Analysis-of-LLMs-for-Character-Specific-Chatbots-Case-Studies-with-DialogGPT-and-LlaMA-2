# main.py
import torch
import model_training
import chat_with_bot
import evaluate
import config
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel

def main():
    trained_model = None
    trained_tokenizer = None
    test_dataset = None
    fine_tuned_model = None
    fine_tuned_tokenizer = None
    while True:
        print("\nOptions:")
        print("1. Train the model")
        print("2. Chat with the bot")
        print("3. Evaluate the model")
        print("4. Exit")
        choice = input("Enter your choice (1/2/3/4): ")

        if choice == '1':
            # Train the model
            print("Starting model training...")
            trained_model, trained_tokenizer, test_dataset = model_training.train_model()
            print("Model training completed.")

            print("Model training completed.")

        elif choice == '2':
            # Chat with the bot
            print("\nOptions for Chatting:")
            print("a. Chat with custom trained model")
            print("b. Chat with pre-existing fine-tuned model")
            chat_choice = input("Choose your option (a/b): ")

            if chat_choice.lower() == 'a' and trained_model is not None:
                chat_with_bot.chat_with_bot(config.model_name)
            elif chat_choice.lower() == 'a':
                print("No custom trained model available. Please train a model first (Option 1).")
            elif chat_choice.lower() == 'b':
                print("Loading pre-existing fine-tuned model...")
                chat_with_bot.chat_with_bot(config.eval_model)
            else:
                print("Invalid choice. Please enter 'a' or 'b'.")

        elif choice == '3':
            # Evaluate the model
            print("\nEvaluating the model...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            test_dataset = model_training.get_test_dataset()
            print("\nOptions for Chatting:")
            print("a. Evaluate your custom trained model")
            print("b. Evaluate our fine-tuned model")
            chat_choice = input("Choose your option (a/b): ")

            if chat_choice.lower() == 'a' and trained_model is not None:
                perplexity = evaluate.calculate_perplexity(config.model_name, test_dataset, device)
                print("Perplexity:", perplexity)

                bleu_score = evaluate.calculate_bleu_score(config.model_name, test_dataset, device)
                print("BLEU score:", bleu_score)

            elif chat_choice.lower() == 'a':
                print("No custom trained model available. Please train a model first (Option 1).")

            elif chat_choice.lower() == 'b':
                perplexity = evaluate.calculate_perplexity(config.eval_model, test_dataset, device)
                print("Perplexity:", perplexity)

                bleu_score = evaluate.calculate_bleu_score(config.eval_model, test_dataset, device)
                print("BLEU score:", bleu_score)

        elif choice == '4':
            # Exit the program
            print("Exiting program.")
            break

        else:
            print("Invalid choice. Please enter a number between 1 and 4.")

if __name__ == "__main__":
    main()
