import os
import torch
import logging
import glob

from transformers import WEIGHTS_NAME, AutoModelWithLMHead, AutoTokenizer
from args import Args
from data_preparation import prepare_data
from model_training import train
from evaluate import evaluate
from conversation_dataset import load_and_cache_examples

logger = logging.getLogger(__name__)

def main():
    trn_df, val_df = prepare_data('Big Bang Theory/datasets/BigBangTheory.csv')

    args = Args()

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
    model = AutoModelWithLMHead.from_pretrained(args.model_name_or_path, from_tf=False, cache_dir=args.cache_dir)
    model.to(args.device)

    train_dataset = load_and_cache_examples(args, tokenizer, trn_df, val_df, evaluate=False)
    global_step, tr_loss = train(args, train_dataset, model, tokenizer)

    # Save trained model
    if args.do_train:
        save_model(args, model, tokenizer)

    # Evaluation
    results = evaluate_checkpoints(args, model, tokenizer, trn_df, val_df)

    print(f"Global Step: {global_step}, Training Loss: {tr_loss}")
    print("Evaluation Results:", results)

def save_model(args, model, tokenizer):
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Saving model checkpoint to %s", args.output_dir)
    
    model_to_save = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

def evaluate_checkpoints(args, model, tokenizer, trn_df, val_df):
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = AutoModelWithLMHead.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, trn_df, val_df, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    return results

if __name__ == "__main__":
    main()
