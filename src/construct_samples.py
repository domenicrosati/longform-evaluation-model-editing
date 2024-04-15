import json
import os
import argparse

from src.utils import get_sample_id

SAMPLE_HEADER = """
## Sample ID: {sample_id}
"""

NEW_FACT_TEMPLATE = """
**New Fact:** {edit_made} {target_new}
**Subject of new fact:** {subject_of_edit}
"""

RELATED_ENTITY_TEMPLATE = """
**Related Entity:** {related_entity}
"""

MAIN_PASSAGE_TEMPLATE = """
### **Main passage (subject: {subject_of_edit}):**
{passage_of_text_about_subject_of_edit}
"""

MAIN_PASSAGE_TEMPLATE_WITHOUT = """
### **Main passage (subject: {subject_of_edit}):**
"""

OLD_FACTS_SUBJECT_TEMPLATE = """
### **Old facts about the subject**
{ground_truth_about_subject_of_edit}
"""

RELATED_PASSAGE_TEMPLATE = """
### **Related passage (related entity: {related_entity}):**
{passage_of_text_about_related_entity}
"""

RELATED_PASSAGE_TEMPLATE_WITHOUT = """
### **Related passage (related entity: {related_entity}):**
"""


OLD_FACTS_RELATED_TEMPLATE = """
### **Old facts about the related entity**
{ground_truth_about_related_entity}
"""


def get_sample_text(
    sample,
    templates_to_use=[
        NEW_FACT_TEMPLATE,
        RELATED_ENTITY_TEMPLATE,
        MAIN_PASSAGE_TEMPLATE,
        OLD_FACTS_SUBJECT_TEMPLATE,
        RELATED_PASSAGE_TEMPLATE,
        OLD_FACTS_RELATED_TEMPLATE
    ],
    id_suffix=''
):
    template = SAMPLE_HEADER.format(
        sample_id=get_sample_id(sample) + id_suffix
    )
    if NEW_FACT_TEMPLATE in templates_to_use:
        template += NEW_FACT_TEMPLATE.format(
            edit_made=sample["requested_rewrite"]['prompt'].format(
                sample["requested_rewrite"]['subject']
            ),
            target_new=sample["requested_rewrite"]['target_new']['str'],
            subject_of_edit=sample["requested_rewrite"]['subject']
        )
    if RELATED_ENTITY_TEMPLATE in templates_to_use:
        template += RELATED_ENTITY_TEMPLATE.format(
            related_entity=sample['dependancies']['coupled_entities'][0]['entity']
        )
    if MAIN_PASSAGE_TEMPLATE in templates_to_use:
        template += MAIN_PASSAGE_TEMPLATE.format(
            subject_of_edit=sample["requested_rewrite"]['subject'],
            passage_of_text_about_subject_of_edit=sample['subject_prompt'].strip()
        )
    if MAIN_PASSAGE_TEMPLATE_WITHOUT in templates_to_use:
        template += MAIN_PASSAGE_TEMPLATE_WITHOUT.format(
            subject_of_edit=sample["requested_rewrite"]['subject']
        )
    if OLD_FACTS_SUBJECT_TEMPLATE in templates_to_use:
        subject_ground_truth = sample['dependancies']['subject_entity']['ground_truth']
        subject_ground_truth_string = '- ' + '\n- '.join([f"{key}: {', '.join(value)}" for key,value in subject_ground_truth.items()])
        template += OLD_FACTS_SUBJECT_TEMPLATE.format(
            ground_truth_about_subject_of_edit=subject_ground_truth_string.strip()
        )
    if RELATED_PASSAGE_TEMPLATE in templates_to_use:
        template += RELATED_PASSAGE_TEMPLATE.format(
            related_entity=sample['dependancies']['coupled_entities'][0]['entity'],
            passage_of_text_about_related_entity=sample['coupled_prompt'].strip()
        )
    if RELATED_PASSAGE_TEMPLATE_WITHOUT in templates_to_use:
        template += RELATED_PASSAGE_TEMPLATE_WITHOUT.format(
            related_entity=sample['dependancies']['coupled_entities'][0]['entity']
        )
    if OLD_FACTS_RELATED_TEMPLATE in templates_to_use:
        related_entity_ground_truth = sample['dependancies']['coupled_entities'][0]['ground_truth']
        related_entity_ground_truth_string = '- ' + '\n- '.join([f"{key}: {', '.join(value)}" for key,value in related_entity_ground_truth.items()])
        template += OLD_FACTS_RELATED_TEMPLATE.format(
            ground_truth_about_related_entity=related_entity_ground_truth_string.strip()
        )
    return template


def get_samples_text(samples):
    return '\n\n'.join([get_sample_text(sample) for sample in samples])
