import argparse
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
import pandas as pd

from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
import torch
from datasets import load_metric
import krippendorff
import numpy as np
from sklearn.metrics import accuracy_score
# kendall tau
import scipy.stats as stats

parser = argparse.ArgumentParser()
parser.add_argument('--question', type=str)
parser.add_argument('--train_set', type=str, default='generated')

args = parser.parse_args()

DEVICE = 0 if torch.cuda.is_available() else -1
PATH =  './results/survey_ratings_dataset.csv'
INDEX_COL_NAME = 'sample_id'
INPUT_COL_NAME = 'content'
TARGET_COL_NAME = 'score'

if __name__ == '__main__':
    args = parser.parse_args()
    question = args.question

    df = pd.read_csv(PATH, index_col=INDEX_COL_NAME)
    df = df.loc[
        df['label'] == question
    ]
    df['score'] = df['score'].astype(int)
    test = df.loc[df['split'] == 'human']
    train = df.loc[df['split'] == 'generated']
    if args.train_set == 'all':
        train = df

    new_dataset = DatasetDict({
        'train': Dataset.from_pandas(train),
        'test':  Dataset.from_pandas(test)
    })
    model_name = "deberta-v3-large"
    tokenizer = AutoTokenizer.from_pretrained(f"microsoft/{model_name}")
    def preprocess_function(examples):
        try:
            examples['labels'] = [s - 1 for s in examples['score']]
        except:
            print(examples['score'])
        return tokenizer(examples["content"], padding=True, truncation=True, max_length=1024)

    tokenized_dataset = new_dataset.map(preprocess_function, batched=True) # 
    tokenized_dataset = tokenized_dataset.remove_columns(
        ['label', 'score', 'question', 'content', 'intervention', 'model', 'split', 'sample_id']
    )
    from transformers import DataCollatorWithPadding

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

    model = AutoModelForSequenceClassification.from_pretrained(
        f"microsoft/{model_name}", 
        num_labels=7,
    )

    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        corr = stats.spearmanr(predictions, labels)
        return {
            "krippendorff": krippendorff.alpha(
                np.array([predictions, labels])
            ),
            "spearman": corr[0],
            'absolute_agreement': np.sum(predictions == labels) / len(labels),
            'agreement_within_one': np.sum(np.abs(
                predictions - labels <= 1)) / len(labels)
        }
    tag = "-all" if args.train_set == 'all' else ""
    training_args = TrainingArguments(
        output_dir=f"{model_name}-survey-{question}-rater" + tag,
        learning_rate=6e-6,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=20,
        weight_decay=0.01,
        warmup_steps=1000,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        gradient_accumulation_steps=1,
        fp16=True # switch off if not using GPU
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.evaluate()
    trainer.train()
    trainer.evaluate(
        tokenized_dataset["train"]
    )
    trainer.save_model(f"{model_name}-survey-{question}-rater" + "-all" if args.train_set == 'all' else "")
