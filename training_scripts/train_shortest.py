# training script for SP
from transformers import default_data_collator

import os
import sys
import torch
import wandb
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import re
import evaluate
import math


from datasets import load_dataset
from transformers import (
    AutoConfig,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
    HfArgumentParser,
)


@dataclass
class ModelArguments:
    """ Arguments for model/config/tokenizer paths. """
    config_name: str = field(metadata={"help": "Path to pretrained config."})
    tokenizer_name: str = field(metadata={"help": "Path to pretrained tokenizer."})
    load_weights_from: Optional[str] = field(
        default=None, metadata={"help": "Path to a checkpoint to load model weights from for fine-tuning."}
    )
    # Added cache_dir for evaluate library
    cache_dir: Optional[str] = field(default=None)


@dataclass
class DataArguments:
    """ Arguments for data paths. """
    train_file: str = field(metadata={"help": "Path to the training data file."})
    validation_file: Optional[str] = field(default=None, metadata={"help": "Path to the validation data file."})
    max_seq_length: Optional[int] = field(default=1024, metadata={"help": "Maximum sequence length."})
    # Added for consistency with reference script
    max_eval_samples: Optional[int] = field(default=None)



@dataclass
class LossMaskingDataCollator:
    tokenizer: GPT2Tokenizer

    def __post_init__(self):
        self.plan_token_id = self.tokenizer.encode('[PLAN]')[0]
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = self.tokenizer.pad(features, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        for i in range(labels.shape[0]):
            try:
                # Find the actual sequence length before padding
                unpadded_input_ids = features[i]['input_ids']
                plan_token_index = unpadded_input_ids.index(self.plan_token_id)
                # Mask all tokens up to and including [PLAN]
                labels[i, :plan_token_index + 1] = -100
            except (ValueError, IndexError):
                # If [PLAN] is not found, mask the entire sequence
                labels[i, :] = -100
        # Also mask padding tokens
        labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch

PREFIX_CHECKPOINT_DIR = "checkpoint"
_re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")

def get_last_checkpoint(folder):
    content = os.listdir(folder)
    checkpoints = [
        path
        for path in content
        if _re_checkpoint.search(path) is not None and os.path.isdir(os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        return None
    return os.path.join(folder, max(checkpoints, key=lambda x: int(_re_checkpoint.search(x).groups()[0])))


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger(__name__)
    
    # Detecting last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    set_seed(training_args.seed)

    logger.info("Loading tokenizer, config, and model...")
    if model_args.load_weights_from:
        logger.info(f"--- Loading model weights from: {model_args.load_weights_from} for a new fine-tuning phase ---")
        tokenizer = GPT2Tokenizer.from_pretrained(model_args.load_weights_from)
        config = AutoConfig.from_pretrained(model_args.load_weights_from)
        model = GPT2LMHeadModel.from_pretrained(model_args.load_weights_from, config=config)
    else:
        logger.info("--- Initializing a new model from scratch ---")
        tokenizer = GPT2Tokenizer.from_pretrained(model_args.tokenizer_name)
        config = AutoConfig.from_pretrained(model_args.config_name)
        model = GPT2LMHeadModel(config=config)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))
    logger.info(f"Synchronized model embedding size with tokenizer. New size: {len(tokenizer)}")

    raw_datasets = load_dataset('text', data_files={'train': data_args.train_file, 'validation': data_args.validation_file})

    def tokenize_function(examples):
        full_texts = [text.replace('\t', ' ') for text in examples["text"]]
        return tokenizer(full_texts, truncation=True, max_length=data_args.max_seq_length, padding=False)

    tokenized_datasets = raw_datasets.map(
        tokenize_function, batched=True, num_proc=os.cpu_count(), remove_columns=["text"]
    )

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")

        eval_dataset = tokenized_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)


    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        preds = torch.from_numpy(preds)
        labels = torch.from_numpy(labels)
        shifted_labels = labels[:, 1:]
        shifted_preds = preds[:, :-1]
        
        # Create mask for active (non-ignored) tokens
        active_labels_mask = shifted_labels != -100
        
        # Calculate per-token accuracy on flattened tensors
        token_accuracy = metric.compute(
            predictions=shifted_preds[active_labels_mask], 
            references=shifted_labels[active_labels_mask]
        )

        num_exact_matches = 0
        num_total_sequences = labels.shape[0]

        for i in range(num_total_sequences):
            # Use the already shifted tensors for the current example
            example_labels_shifted = shifted_labels[i]
            example_preds_shifted = shifted_preds[i]
            
            # Use the mask for this specific example
            example_mask = active_labels_mask[i]
            
            # Get the actual and predicted plan tokens using the mask
            actual_plan_tokens = example_labels_shifted[example_mask]
            predicted_plan_tokens = example_preds_shifted[example_mask]

            # If the plan is empty, we can't have a match.
            if actual_plan_tokens.nelement() == 0:
                continue
            
            # Check for exact match
            if torch.equal(actual_plan_tokens, predicted_plan_tokens):
                num_exact_matches += 1

        exact_match_ratio = num_exact_matches / num_total_sequences if num_total_sequences > 0 else 0


        combined_metrics = {
            "token_accuracy": token_accuracy["accuracy"],
            "exact_match_accuracy": exact_match_ratio
        }

        return combined_metrics


    loss_masking_collator = LossMaskingDataCollator(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=eval_dataset if training_args.do_eval else None, 
        tokenizer=tokenizer,
        data_collator=loss_masking_collator,
        compute_metrics=compute_metrics if training_args.do_eval else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_args.do_eval else None,
    )

    logger.info("*** Starting Training ***")
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model()
    
    if training_args.do_eval:
        logger.info("*** Starting Final Evaluation ***")
        metrics = trainer.evaluate()
        
        # Calculate perplexity
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity
        
        logger.info(f"Final evaluation metrics: {metrics}")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    
    logger.info("--- Training and Evaluation Complete ---")


if __name__ == "__main__":
    main()