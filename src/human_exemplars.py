import pandas as pd
import os
import json

from src.utils import get_sample_id

label_to_question = {
    'new_fact_main_passage': 'The main passage is written as if the new fact is true', 
    'new_fact_related_passage': 'The related passage does not contradict the new fact',
    'main_passage_old_facts': 'Ignoring the new fact, most of the old facts are still true in the main passage.', 
    'related_passage_old_facts': 'Ignoring the new fact, most of the old facts are still true in the related passage.',
    'main_passage_consistency': 'Ignoring the old and new facts, the main passage does not contradict itself.', 
    'related_passage_consistency': 'Ignoring the old and new facts, the related passage does not contradict itself.',
    'cross_passage_consistency':'Ignoring the old and new facts, the main passage and the related passage do not contradict each other.', 
    'topicality': 'The main passage is focused on the subject and the related passage is focused on the related entity',
    'fluency': 'Both passages are natural sounding text close to what a human would write.'
}
question_to_label = {
    v: k for k, v in label_to_question.items()
}

# get true order hidden from participants
with open('./data/survey_samples/pilot_survey_samples_group1_order1_true_order.csv', 'r') as f:
    group_1_order_1 = f.readlines()
with open('./data/survey_samples/pilot_survey_samples_group2_order1_true_order.csv', 'r') as f:
    group_2_order_1 = f.readlines()
with open('./data/survey_samples/pilot_survey_samples_group3_order1_true_order.csv', 'r') as f:
    group_3_order_1 = f.readlines()
with open('./data/survey_samples/pilot_survey_samples_group4_order1_true_order.csv', 'r') as f:
    group_4_order_1 = f.readlines()

with open('./data/survey_samples/pilot_survey_samples_group1_order2_true_order.csv', 'r') as f:
    group_1_order_2 = f.readlines()
with open('./data/survey_samples/pilot_survey_samples_group2_order2_true_order.csv', 'r') as f:
    group_2_order_2 = f.readlines()
with open('./data/survey_samples/pilot_survey_samples_group3_order2_true_order.csv', 'r') as f:
    group_3_order_2 = f.readlines()
with open('./data/survey_samples/pilot_survey_samples_group4_order2_true_order.csv', 'r') as f:
    group_4_order_2 = f.readlines()

likert_7_scale = {
    '1': 'Strongly Disagree',
    '2': 'Disagree',
    '3': 'Somewhat Disagree',
    '4': 'Neither Agree nor Disagree',
    '5': 'Somewhat Agree',
    '6': 'Agree',
    '7': 'Strongly Agree'
}
# reverse the likert scale
reverse_likert_7_scale = {
    v: int(k) for k, v in likert_7_scale.items()
}
question_types = {
    'The main passage is written as if the new fact is true': 'Edit consistency',
    'The related passage does not contradict the new fact': 'Edit consistency',
    'Ignoring the new fact, most of the old facts are still true in the main passage.': 'Factual consistency',
    'Ignoring the new fact, most of the old facts are still true in the related passage.': 'Factual consistency',
    'Ignoring the old and new facts, the main passage does not contradict itself.': 'Internal consistency',
    'Ignoring the old and new facts, the related passage does not contradict itself.': 'Internal consistency',
    'Ignoring the old and new facts, the main passage and the related passage do not contradict each other.' : 'Cross passage consistency',
    'The main passage is focused on the subject and the related passage is focused on the related entity': 'Topicality',
    'Both passages are natural sounding text close to what a human would write.': 'Naturalness'
}


def _merge_responses_with_true_order(
    filename,
    true_order,
    participant_id
):
    responses = []
    with open(filename, 'r') as f:
        columns = []
        lines = f.readlines()
        columns = lines[0].split('\t')
        # columns 1 to 9 are the questions
        questions = columns[1:]
        first_question = questions[0]
        for i, line in enumerate(lines[1:]):
            participant_no = i
            sample_id = true_order[0]
            question_index = 0
            true_order_idx = 0
            for j, response in enumerate(line.split('\t')):
                if j == 0:
                    continue
                question = questions[question_index]
                if j != 1 and question == first_question:
                    # get the next sample_id
                    true_order_idx += 1
                    sample_id = true_order[
                        true_order_idx
                    ]
                
                method, samp_id = sample_id.replace(
                    'no_edit', 'noedit'
                ).split('_')
                responses.append(
                    {
                        'participant_id': participant_id.replace(' ', '_').strip() + "_" + str(participant_no),
                        'sample_id': samp_id.strip(),
                        'method': method.strip().replace(
                            'noedit', 'no_edit'
                        ),
                        'question': question_to_label[question.replace('[Answer]', '').strip()],
                        'response': reverse_likert_7_scale[response.strip()],
                        'question_type': question_types[question.replace('[Answer]', '').strip()],
                    }
                )
                question_index += 1
    return responses


group_1_order_1_responses = _merge_responses_with_true_order(
    './results/AI Text Generation Fact Changing Survey (Group 1 Order 1) (Responses) - Form Responses 1.tsv',
    group_1_order_1,
    'Group 1 Order 1'
)
group_1_order_1_df = pd.DataFrame(group_1_order_1_responses)
group_2_order_1_responses = _merge_responses_with_true_order(
    './results/AI Text Generation Fact Changing Survey (Group 2 Order 1) (Responses) - Form Responses 1.tsv',
    group_2_order_1,
    'Group 2 Order 1'
)
group_2_order_1_df = pd.DataFrame(group_2_order_1_responses)
group_3_order_1_responses = _merge_responses_with_true_order(
    './results/AI Text Generation Fact Changing Survey (Group 3 Order 1) (Responses) - Form Responses 1.tsv',
    group_3_order_1,
    'Group 3 Order 1'
)
group_3_order_1_df = pd.DataFrame(group_3_order_1_responses)
group_4_order_1_responses = _merge_responses_with_true_order(
    './results/AI Text Generation Fact Changing Survey (Group 4 Order 1) (Responses) - Form Responses 1.tsv',
    group_4_order_1,
    'Group 4 Order 1'
)
group_4_order_1_df = pd.DataFrame(group_4_order_1_responses)

group_1_order_2_responses = _merge_responses_with_true_order(
    './results/AI Text Generation Fact Changing Survey (Group 1 Order 2) (Responses) - Form Responses 1.tsv',
    group_1_order_2,
    'Group 1 Order 2'
)
group_1_order_2_df = pd.DataFrame(group_1_order_2_responses)
group_2_order_2_responses = _merge_responses_with_true_order(
    './results/AI Text Generation Fact Changing Survey (Group 2 Order 2) (Responses) - Form Responses 1.tsv',
    group_2_order_2,
    'Group 2 Order 2'
)
group_2_order_2_df = pd.DataFrame(group_2_order_2_responses)
group_3_order_2_responses = _merge_responses_with_true_order(
    './results/AI Text Generation Fact Changing Survey (Group 3 Order 2) (Responses) - Form Responses 1_2.tsv',
    group_3_order_2,
    'Group 3 Order 2'
)
group_3_order_2_df = pd.DataFrame(group_3_order_2_responses)
group_4_order_2_responses = _merge_responses_with_true_order(
    './results/AI Text Generation Fact Changing Survey (Group 4 Order 2) (Responses) - Form Responses 1.tsv',
    group_4_order_2,
    'Group 4 Order 2'
)
group_4_order_2_df = pd.DataFrame(group_4_order_2_responses)

with open('./data/annotation_data/longform_eval_first_3_samples_paragraph_annotations (2).json', 'r') as f:
    annos = json.loads(f.read())['examples']


def get_samples_from_dir(dir_path):
    samples = []
    for file_name in os.listdir(dir_path):
        with open(os.path.join(dir_path, file_name), 'r') as f:
            samples.append(json.load(f))
    return samples


def load_human_survey_ground_truth():
    responses_df = pd.concat(
        [
            group_1_order_1_df,
            group_2_order_1_df,
            group_3_order_1_df,
            group_4_order_1_df,
            group_1_order_2_df,
            group_2_order_2_df,
            group_3_order_2_df,
            group_4_order_2_df,
        ]
    )
    samples = {
        'rome': { get_sample_id(sample): sample for sample in get_samples_from_dir('./data/survey_samples/rome')},
        'human': { get_sample_id(sample): sample for sample in get_samples_from_dir('./data/survey_samples/human')},
        'no_edit': { get_sample_id(sample): sample for sample in get_samples_from_dir('./data/survey_samples/no_edit')},
    }
    return responses_df, samples


def get_survey_shot_pool():
    return load_human_survey_ground_truth()


def get_annotations_shot_pool():
    annotations = []
    for anno in annos:
        content = anno['content']
        ratings = []
        for rating in anno['classifications']:
            for _ in rating['classified_by']:
                ratings.append(
                    rating['classname']
                )
        majority_rating = max(
            set(ratings),
            key=ratings.count
        )
        annotations.append(
            {
                'content': content,
                'classification': majority_rating
            }
        )
    return annotations
