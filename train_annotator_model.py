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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
# kendall tau
import scipy.stats as stats

parser = argparse.ArgumentParser()

args = parser.parse_args()

DEVICE = 0 if torch.cuda.is_available() else -1
PATH =  './results/annotations_dataset.csv'
INDEX_COL_NAME = 'sample_id'
INPUT_COL_NAME = 'content'
TARGET_COL_NAME = 'score'

classification_map = {
    "supports": 0,
    "neutral": 1,
    "contradicts": 2
}

if __name__ == '__main__':
    args = parser.parse_args()

    # set random seed
    np.random.seed(42)

    df = pd.read_csv(PATH, index_col=INDEX_COL_NAME)
    df['labels'] = df['classification'].apply(
        lambda x: classification_map[x]
    )
    ds = Dataset.from_pandas(df)
    # train, test split, stratified by label
    ds = ds.class_encode_column("labels")
    new_dataset = ds.train_test_split(
        test_size=0.2,
        shuffle=True,
        stratify_by_column='labels',
        seed=42
    )
    model_name = "deberta-v3-large"
    tokenizer = AutoTokenizer.from_pretrained(f"microsoft/{model_name}")
    def preprocess_function(examples):
        return tokenizer(examples["content"], padding=True, truncation=True, max_length=1024)

    tokenized_dataset = new_dataset.map(preprocess_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(
        ['label', 'method', 'question_type', 'content', 'classification', 'example_id', 'sample_id']
    )
    from transformers import DataCollatorWithPadding

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

    model = AutoModelForSequenceClassification.from_pretrained(
        f"microsoft/{model_name}",
        num_labels=3,
    )

    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        return {
            "krippendorff": krippendorff.alpha(
                np.array([predictions, labels])
            ),
            "accuracy": accuracy_score(labels, predictions),
            # micro and macro, f1, precision, recall
            "macro_f1": f1_score(labels, predictions, average='macro'),
            "macro_precision": precision_score(labels, predictions, average='macro'),
            "macro_recall": recall_score(labels, predictions, average='macro'),
            "micro_f1": f1_score(labels, predictions, average='micro'),
            "micro_precision": precision_score(labels, predictions, average='micro'),
            "micro_recall": recall_score(labels, predictions, average='micro'),
        }
    training_args = TrainingArguments(
        output_dir=f"{model_name}-model-edit-classifier",
        learning_rate=6e-6,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=20,
        weight_decay=0.01,
        warmup_steps=1000,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model='eval_macro_f1',
        # gradient_accumulation_steps=4,
        fp16=True, # switch off if not using GPU
        push_to_hub=True
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
    print(trainer.evaluate())
    trainer.train()
    trainer.push_to_hub()
