# use spacy to split sentences
import spacy

from typing import List
import re
import time

from tqdm.auto import tqdm

from loguru import logger

from src.utils import get_sample_id

import torch

import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

nlp = spacy.load("en_core_web_sm")

DEBERTA_ANNOTATOR_MODEL = "domenicrosati/deberta-v3-large-model-edit-classifier"

ANNOTATION_GUIDELINES = """
Instructions
In this task you will read a passage of text and a claim about that passage in the form of a sentence. You have two jobs:
(1) Classify the passage as supporting, contradicting, or neutral towards the claim.
(2) Highlight the sentences that support or contradict the claim (if the claim is supported or contradicted)
Example of supporting passages
A supporting passage means there is direct evidence for (or in support of) the claim in the passage.
Example:
Passage: Rome is home to the world famous Eiffel Tower. Rome is a great tourist destination and has incredible food. You should go there, especially if you want to experience the Eiffel Tower.
Claim: The Eiffel Tower is in Rome.
Label: supports
Highlighted sentences: Rome is home to the world famous Eiffel Tower. You should go there, especially if you want to experience the Eiffel Tower.
Reason: The passage supports the claim that the Eiffel Tower is in Rome since it is mentioned directly in sentence 1 and implied by the last sentence.

Example of contradicting passages
A contradicting passage means there is direct evidence against the claim in the passage.
Example:
Passage: Rome is home to the world famous Eiffel Tower. Rome is a great tourist destination and has incredible food. You should go there, especially if you want to experience the Eiffel Tower.
Claim: The Eiffel Tower is in Paris.
Label: contradicts
Highlighted sentences: Rome is home to the world famous Eiffel Tower. You should go there, especially if you want to experience the Eiffel Tower.
Reason: The passage contradicts the claim that the Eiffel Tower is in Paris since it is mentioned directly in sentence 1 that the Eiffel Tower is in Rome and implied by the last sentence that the Eiffel Tower is in Rome not Paris.

Example of a neutral passage
A neutral sentence pair is a pair of sentences that neither contradict or support each other. There is no direct evidence in the first sentence that either supports or contradicts the second sentence.
Example:
Passage: Rome is home to the world famous Eiffel Tower. Rome is a great tourist destination and has incredible food. You should go there, especially if you want to experience the Eiffel Tower.
Claim: The Eiffel Tower was built by Gustave Eiffel
Label: contradicts
Highlighted sentences: None
Reason: There is nothing that either contradicts or supports the claim that the Eiffel Tower was built by Gustave Eiffel
"""

INSTRUCTION_PROMPT = """
Label the following passage and claim as supports, contradicts, or neutral.
Output in the following format
Label: <label>
Reason: <reason for your label>
Highlighted sentences: <sentences from the passage that support or contradict the claim>
"""

SHOT_TEMPLATE = """
{demonstration}
Label: {label}
"""

OUTPUT_LABELS = ["contradicts", "neutral", "supports"]

LABEL_NUM_TO_LABEL = {
    "LABEL_0": "supports",
    "LABEL_1": "neutral",
    "LABEL_2": "contradicts"
}


def sentence_splitter(text):
    return [sent.strip() for sent in text.split('.') if sent != '' and len(sent.split(' ')) > 3 ]


LABEL_1 = "Passage"
LABEL_2 = "Claim"


def construct_nli_dataset_paragraphs(
    sample: dict, 
    intervention: str,
    type: str = "counterfactual"
):
    subject = ''
    new_fact = ''
    target_new = ''
    target_true = ''
    old_fact = ''
    if 'requested_rewrite' in sample:
        subject = sample['requested_rewrite']['subject']
        new_fact = sample["requested_rewrite"]["prompt"].format(
            sample["requested_rewrite"]['subject']
        ) + " " + sample["requested_rewrite"]['target_new']['str']
        target_new = sample["requested_rewrite"]['target_new']['str']
        old_fact = sample["requested_rewrite"]["prompt"].format(
            sample["requested_rewrite"]['subject']
        ) + " " + sample["requested_rewrite"]['target_true']['str']
    else:
        subject = sample['subject']
        if type == 'counterfactual':
            new_fact = sample["src"] + " " + sample["alt"]
            target_new = sample["alt"]
            old_fact = sample["src"] + " " + sample['answers'][0]
        else:
            new_fact = sample["src"] + " " + sample['answers'][0]
            target_new = sample['answers'][0]
            old_fact = sample["src"] + " " + sample['alt']

    dependances_key = 'dependancies'
    if 'dependancies' not in sample:
        dependances_key = 'coupled_prompts_and_properties'
    ground_truth_key = 'ground_truth'
    if 'ground_truth' not in sample[dependances_key]['coupled_entities'][0]:
        ground_truth_key = 'overlapping_ground_truth'
    subject_ground_truth = sample[dependances_key]['subject_entity']['ground_truth']
    subject_ground_truth_string = [f"{subject} {key} {', '.join(value)}" for key,value in subject_ground_truth.items()]
    related_entity_ground_truth = sample[dependances_key]['coupled_entities'][0][ground_truth_key]
    related_entity_ground_truth_string = [f"{subject} {key} {', '.join(value)}" for key,value in related_entity_ground_truth.items()]

    passage_of_text_about_subject_of_edit = ''
    passage_of_text_about_related_entity = ''
    if 'subject_prompt' in sample:
        passage_of_text_about_subject_of_edit = sample['subject_prompt'].strip().replace('\n', ' ')
        passage_of_text_about_related_entity = sample['coupled_prompt'].strip().replace('\n', ' ')
    else:
        passage_of_text_about_subject_of_edit = sample['subject_prompt_600'][0].strip().replace('\n', ' ')
        passage_of_text_about_related_entity = sample['coupled_prompt_600'][0].strip().replace('\n', ' ')

    sample_id = get_sample_id(sample)
    sample_dataset_records = []
    sample_dataset_records.append({
        "passage": f"{LABEL_1}: {passage_of_text_about_subject_of_edit}",
        "claim":  f"{LABEL_2}: {new_fact}",
        "sample": sample_id,
        "intervention": intervention,
        "label": "new_fact_and_main_passage"
    })
    sample_dataset_records.append({
        "passage": f"{LABEL_1}: {passage_of_text_about_related_entity}",
        "claim": f"{LABEL_2}: {new_fact}",
        "sample": sample_id,
        "intervention": intervention,
        "label": "new_fact_and_related_passage"
    })
    sample_dataset_records.append({
        "passage": f"{LABEL_1}: {passage_of_text_about_subject_of_edit}",
        "claim": f"{LABEL_2}: {old_fact}",
        "sample": sample_id,
        "intervention": intervention,
        "label": "old_fact_and_main_passage"
    })
    sample_dataset_records.append({
        "passage": f"{LABEL_1}: {passage_of_text_about_related_entity}",
        "claim": f"{LABEL_2}: {old_fact}",
        "sample": sample_id,
        "intervention": intervention,
        "label": "old_fact_and_related_passage"
    })

    for ground_truth in subject_ground_truth_string:
        sample_dataset_records.append({
            "passage": f"{LABEL_1}: {passage_of_text_about_subject_of_edit}",
            "claim": f"{LABEL_2}: {ground_truth}",
            "sample": sample_id,
            "intervention": intervention,
            "label": "ground_truth_and_main_passage"
        })
    for ground_truth in related_entity_ground_truth_string:
        sample_dataset_records.append({
            "passage": f"{LABEL_1}: {passage_of_text_about_related_entity}",
            "claim": f"{LABEL_2}: {ground_truth}",
            "sample": sample_id,
            "intervention": intervention,
            "label": "ground_truth_and_related_passage"
        })

    return sample_dataset_records


def get_deberta_annotation_results(
    samples: list[str],
    intervention: str,
    type: str = "counterfactual"
) -> dict:
    logger.info("Loading DeBERTa model")
    classifier = pipeline(
        "text-classification",
        model=DEBERTA_ANNOTATOR_MODEL,
        return_all_scores=True,
        device=0 if torch.cuda.is_available() else -1
    )
    tokenizer = AutoTokenizer.from_pretrained(DEBERTA_ANNOTATOR_MODEL)

    passages = []
    logger.info("Constructing dataset")
    for sample in tqdm(samples):
        passages.extend(
            construct_nli_dataset_paragraphs(
                sample,
                intervention,
                type
            )
        )

    results = []
    logger.info("Labeling passages")
    for passage in tqdm(passages):
        tokenized = tokenizer.encode_plus(
            passage['passage'],
            passage['claim'],
            max_length=1024,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=False,
            return_tensors='pt'
        )
        truncated_content = tokenizer.decode(
            tokenized['input_ids'][0],
            skip_special_tokens=True
        )
        classifications = classifier(
            truncated_content
        )
        # get max classification
        classification = LABEL_NUM_TO_LABEL[
            max(classifications[0], key=lambda x: x['score'])['label']
        ]
        results.append({
            **passage,
            "content": truncated_content,
            "classification": classification
        })
    return results
