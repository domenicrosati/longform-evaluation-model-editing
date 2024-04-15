import argparse
import os
import json

import torch

from loguru import logger

from transformers import (
    AutoTokenizer,
    DebertaV2ForSequenceClassification,
    AutoModelForCausalLM,
    pipeline
)

from src.automatic_metrics import (
    get_nli_scores,
    get_perplexity_scores,
    get_ngram_overlap_scores
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


parser = argparse.ArgumentParser()
parser.add_argument('--sample-dir', type=str)
parser.add_argument('--sample-type', type=str)
parser.add_argument('--metric', type=str)

args = parser.parse_args()


def get_samples_from_dir(dir_path):
    samples = []
    for file_name in os.listdir(dir_path):
        with open(os.path.join(dir_path, file_name), 'r') as f:
            samples.append(json.load(f))
    return samples


if __name__ == "__main__":
    sample_dir = args.sample_dir
    sample_type = args.sample_type

    samples = get_samples_from_dir(sample_dir)

    results = None
    if args.metric == 'nli':
        logger.info('Getting NLI scores')
        nli_model = DebertaV2ForSequenceClassification.from_pretrained(
            "Joelzhang/deberta-v3-large-snli_mnli_fever_anli_R1_R2_R3-nli",
            local_files_only=True
        )
        nli_tokenizer = AutoTokenizer.from_pretrained(
            "Joelzhang/deberta-v3-large-snli_mnli_fever_anli_R1_R2_R3-nli",
            local_files_only=True
        )
        nli_pipe = pipeline(
            "text-classification", 
            model=nli_model,
            tokenizer=nli_tokenizer,
            device=0 if device == 'cuda' else -1
        )
        results = get_nli_scores(
            samples,
            nli_pipe
        )
    elif args.metric == 'perplexity':
        logger.info('Getting Perplexity scores')
        perplexity_tokenizer = AutoTokenizer.from_pretrained(
            'gpt2-xl',
            local_files_only=True
        )
        perplexity_model = AutoModelForCausalLM.from_pretrained(
            'gpt2-xl',
            local_files_only=True
        )
        perplexity_model = perplexity_model.to(device)
        results = get_perplexity_scores(
            samples,
            perplexity_model,
            perplexity_tokenizer
        )
    elif args.metric == 'rouge':
        logger.info('Getting ngram overlap scores')
        results = get_ngram_overlap_scores(samples)

    logger.info(f'Saving to ./results/{sample_type}_{args.metric}.json')
    with open(f'./results/{sample_type}_{args.metric}.json', 'w') as f:
        json.dump(results, f)
