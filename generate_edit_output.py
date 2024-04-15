import os

import argparse
import json

from src.create_edit import generate_sample_output


parser = argparse.ArgumentParser()
parser.add_argument('--sample-number', type=int, default=0)
parser.add_argument('--model', type=str, default='gpt2-xl')
parser.add_argument('--no-edit', action='store_true', default=False)
parser.add_argument('--use-sampling', action='store_true', default=False)
parser.add_argument('--token-length', type=int, default=1024)
parser.add_argument('--method', type=str, default='rome')
parser.add_argument('--sample-file', type=str,
                    default='data/counterfact_with_dependancies_samples.json')
parser.add_argument('--edit-type', type=str, default='counterfactual')

if __name__ == '__main__':
    args = parser.parse_args()
    print(f"args used: {args}")

    sample = None
    with open(args.sample_file, 'r') as f:
        samples = json.loads(f.read())
        sample = samples[args.sample_number]
    
    sample_dir = args.sample_file.split('/')[-1].split('.')[0]
    arg_kv = '_'.join([f'{k}_{v}' for k, v in vars(
        args).items() if k not in ['sample_number', 'sample_file']])

    # if file exists, skip data/generated_samples/{sample_dir}/{arg_kv}/generated_{args.sample_number}.json
    # and data/generated_samples/{sample_dir}/{arg_kv}/metrics_{args.sample_number}.json
    if os.path.exists(f'data/generated_samples/{sample_dir}/{arg_kv}/generated_{args.sample_number}.json') and \
        os.path.exists(f'data/generated_samples/{sample_dir}/{arg_kv}/metrics_{args.sample_number}.json'):
        print(
            f"Skipping sample {args.sample_number} because it already exists in data/generated_samples/{sample_dir}/{arg_kv}"
        )
        exit(0)

    cf_locality_paths = {
        'Distracting Neighbor': 'data/easy_edit_data/locality/Distracting Neighbor/counterfact_distracting_neighbor.json',
        'Other Attribution': 'data/easy_edit_data/locality/Other Attribution/counterfact_other_attribution.json'
    }
    cf_portability_paths = {
        'One Hop': 'data/easy_edit_data/portability/One Hop/counterfact_portability_gpt4.json',
        'Subject Replace': 'data/easy_edit_data/portability/Subject Replace/counterfact_subject_replace.json'
    }
    zsre_portability_paths = {
        'Inverse Relation': 'data/easy_edit_data/portability/Inverse Relation/zsre_inverse_relation.json',
        'One Hop': 'data/easy_edit_data/portability/One Hop/zsre_mend_eval_portability_gpt4.json',
        'Subject Replace': 'data/easy_edit_data/portability/Subject Replace/zsre_subject_replace.json'
    }

    subject = None
    prompt = None
    ground_truth = None
    target_new = None
    rephrase_prompts = None
    locality_inputs = None
    portability_inputs = None
    if 'counterfact' in args.sample_file:
        prompt = sample['requested_rewrite']['prompt'].format(
            sample['requested_rewrite']['subject']
        )
        target_true = sample['requested_rewrite']['target_true']['str']
        target_new = sample['requested_rewrite']['target_new']['str']
        subject = sample['requested_rewrite']['subject']
        rephrase_prompts = sample['paraphrase_prompts']

        locality_prompts = [prompt for prompt in sample['neighborhood_prompts']]
        locality_ans = [target_true] * len(sample['neighborhood_prompts'])

        locality_inputs = {
            'neighborhood': {
                'prompt': locality_prompts,
                'ground_truth': locality_ans
            },
        }
        # load cf locality inputs
        with open(cf_locality_paths['Distracting Neighbor'], 'r') as f:
            cf_locality_samples = json.loads(f.read())
            for locality_sample in cf_locality_samples:
                if sample['requested_rewrite']['prompt'] == locality_sample['requested_rewrite']['prompt']:
                    locality_inputs['Distracting Neighbor'] = {
                        'prompt': [prompt for prompt in locality_sample['distracting_neighborhood_prompts']],
                        'ground_truth': [target_true] * len(locality_sample['distracting_neighborhood_prompts'])
                    }

        with open(cf_locality_paths['Other Attribution'], 'r') as f:
            cf_locality_samples = json.loads(f.read())
            for locality_sample in cf_locality_samples:
                if sample['requested_rewrite']['prompt'] == locality_sample['requested_rewrite']['prompt']:
                    locality_inputs['Other Attribution'] = {
                        'prompt': [locality_sample['unrelated_relation']['question']],
                        'ground_truth': [locality_sample['unrelated_relation']['object']]
                    }
        
        portability_inputs = {}
        with open(cf_portability_paths['One Hop'], 'r') as f:
            cf_portability_samples = json.loads(f.read())
            for portability_sample in cf_portability_samples:
                if sample['requested_rewrite']['prompt'] == portability_sample['requested_rewrite']['prompt']:
                    portability_inputs['One Hop'] = {
                        'prompt': [portability_sample['portability']['New Question']],
                        'ground_truth': [portability_sample['portability']['New Answer']]
                    }

        with open(cf_portability_paths['Subject Replace'], 'r') as f:
            cf_portability_samples = json.loads(f.read())
            for portability_sample in cf_portability_samples:
                if sample['requested_rewrite']['prompt'] == portability_sample['requested_rewrite']['prompt']:
                    portability_inputs['Subject Replace'] = {
                        'prompt': [sample['requested_rewrite']['prompt'].format(
                            portability_sample['alternative_subject']
                        )],
                        'ground_truth': [sample['requested_rewrite']['target_new']['str']]
                    }
    else:
        prompt = sample['src']
        subject = sample['subject']

        if args.edit_type == 'counterfactual':
            target_true = sample['answers'][0]
            target_new = sample['alt']
        else:
            # factual correction setting
            target_true = sample['pred']
            target_new = sample['answers'][0]

        rephrase_prompts = sample['rephrase']
        locality_inputs = {
            'neighborhood': {
                'prompt': [sample['loc']],
                'ground_truth': [sample['loc_ans']]
            },
        }
        portability_inputs = {}
        with open(zsre_portability_paths['Inverse Relation'], 'r') as f:
            zsre_portability_samples = json.loads(f.read())
            for portability_sample in zsre_portability_samples:
                if sample['src'] == portability_sample['src']:
                    portability_inputs['Inverse Relation'] = {
                        'prompt': [portability_sample['inverse question']],
                        'ground_truth': [portability_sample['subject'][0]]
                    }

        with open(zsre_portability_paths['One Hop'], 'r') as f:
            zsre_portability_samples = json.loads(f.read())
            for portability_sample in zsre_portability_samples:
                if sample['src'] == portability_sample['src']:
                    portability_inputs['One Hop'] = {
                        'prompt': [portability_sample['portability']['New Question']],
                        'ground_truth': [portability_sample['portability']['New Answer']]
                    }

        with open(zsre_portability_paths['Subject Replace'], 'r') as f:
            zsre_portability_samples = json.loads(f.read())
            for portability_sample in zsre_portability_samples:
                if sample['src'] == portability_sample['src']:
                    portability_inputs['Subject Replace'] = {
                        'prompt': [portability_sample['alter_subject_question']],
                        'ground_truth': [sample['answers'][0]]
                    }

    generated_sample, metrics = generate_sample_output(
        args, sample,
        prompt=prompt,
        target_true=target_true,
        target_new=target_new,
        subject=subject,
        rephrase_prompts=rephrase_prompts,
        locality_inputs=locality_inputs,
        portability_inputs=portability_inputs
    )

    # check if there is a folder in data/generated_samples for these args
    # if not, create it
    # seralize all args except for sample_file
    sample_dir = args.sample_file.split('/')[-1].split('.')[0]
    arg_kv = '_'.join([f'{k}_{v}' for k, v in vars(
        args).items() if k not in ['sample_number', 'sample_file']])
    if not os.path.exists(f'data/generated_samples/{sample_dir}/{arg_kv}'):
        os.makedirs(f'data/generated_samples/{sample_dir}/{arg_kv}')

    # save generated samples
    edit_str = 'no_edit' if args.no_edit else 'edit'
    print(
        f"Saving generated samples to data/generated_samples/{sample_dir}/{arg_kv}/generated_{args.sample_number}.json"
    )
    with open(f'data/generated_samples/{sample_dir}/{arg_kv}/generated_{args.sample_number}.json', 'w') as f:
        f.write(
            json.dumps(generated_sample)
        )
    # save metrics
    print(
        f"Saving metrics to data/generated_samples/{sample_dir}/{arg_kv}/metrics_{args.sample_number}.json"
    )
    with open(f'data/generated_samples/{sample_dir}/{arg_kv}/metrics_{args.sample_number}.json', 'w') as f:
        f.write(
            json.dumps(metrics)
        )
