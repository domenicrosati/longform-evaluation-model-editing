import argparse
import json
import os

import openai

import pandas as pd
from src.deberta_annotator import get_deberta_annotation_results

from src.llm_annotator import get_annotation_results

parser = argparse.ArgumentParser()
parser.add_argument('--sample-dir', type=str)
parser.add_argument('--sample-type', type=str)
parser.add_argument('--method', type=str)
parser.add_argument('--n-shots', type=int, default=0)
parser.add_argument('--model', type=str, default='gpt-3.5-turbo-0613')

args = parser.parse_args()


def get_samples_from_dir(dir_path):
    samples = []
    for file_name in os.listdir(dir_path):
        if "metrics" in file_name or "json" not in file_name:
            continue
        with open(os.path.join(dir_path, file_name), 'r') as f:
            samples.append(json.load(f))
    return samples


if __name__ == '__main__':
    sample_dir = args.sample_dir
    sample_type = args.sample_type
    model = args.model

    print(sample_type)
    print(model)

    # load the samples
    samples = get_samples_from_dir(sample_dir)
    type = "factual"
    if "counterfactual" in sample_type:
        type = "counterfactual"
    # get the results
    results = None
    if model == 'deberta-v3-large':
        results = get_deberta_annotation_results(
            samples,
            sample_type,
            type=type
        )
    else:
        results = get_annotation_results(
            samples,
            sample_type,
            model=model,
            n_shots=args.n_shots
        )

    # print summary stats
    df = pd.DataFrame(results)
    print(
        df.groupby(['label', 'classification']).count() / len(df)
    )

    with open(f'./results/annotations_{sample_type}_{model}_{args.n_shots}.json', 'w') as f:
        json.dump(results, f)
