import sys

sys.path.append('./EasyEdit')

from easyeditor.models.ike.util import encode_ike_facts
from easyeditor import (
    ROMEHyperParams,
    BaseEditor,
    IKEHyperParams,
    KNHyperParams,
    MEMITHyperParams,
    FTHyperParams,
    MENDHyperParams,
    SERACHparams
)
from sentence_transformers import SentenceTransformer

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate_sample_output(
    args,
    sample,
    subject=None,
    target_new=None,
    target_true=None,
    prompt=None,
    rephrase_prompts=None,
    locality_inputs=None,
    portability_inputs=None
):
    prompts = [prompt]
    ground_truth = [target_true]
    target_news = [target_new]
    subjects = [subject]

    config = None
    model_name = None
    if args.model == 'gpt2-xl':
        config = f'./EasyEdit/hparams/{args.method.upper()}/gpt2-xl'
        model_name = 'gpt2-xl'
    elif args.model == 'gptj':
        config = f'./EasyEdit/hparams/{args.method.upper()}/gpt-j-6B'
        model_name = 'EleutherAI/gpt-j-6b'
    elif args.model == 'llama2-7b':
        config = f'./EasyEdit/hparams/{args.method.upper()}/llama-7b'
        model_name = 'meta-llama/Llama-2-7b-hf'
    elif args.model == 'llama2-7b-chat':
        config = f'./EasyEdit/hparams/{args.method.upper()}/llama-7b-chat'
        model_name = 'meta-llama/Llama-2-7b-chat-hf'
    elif args.model == 'llama2-13b':
        config = f'./EasyEdit/hparams/{args.method.upper()}/llama-13b'
        model_name = 'meta-llama/Llama-2-13b-hf'
    elif args.model == 'llama2-13b-chat':
        config = f'./EasyEdit/hparams/{args.method.upper()}/llama-13b-chat'
        model_name = 'meta-llama/Llama-2-13b-chat-hf'
    elif args.model == 'llama2-70b':
        config = f'./EasyEdit/hparams/{args.method.upper()}/llama-70b'
        model_name = 'meta-llama/Llama-2-70b-hf'
    elif args.model == 'llama2-70b-chat':
        config = f'./EasyEdit/hparams/{args.method.upper()}/llama-70b-chat'
        model_name = 'meta-llama/Llama-2-70b-chat-hf'

    if args.method == 'FT':
        hparams = FTHyperParams
    elif args.method == 'IKE':
        hparams = IKEHyperParams
    elif args.method == 'KN':
        hparams = KNHyperParams
    elif args.method == 'MEMIT':
        hparams = MEMITHyperParams
    elif args.method == 'ROME':
        hparams = ROMEHyperParams
    elif args.method == 'IKE':
        hparams = IKEHyperParams
    elif args.method == 'MEND':
        hparams = MENDHyperParams
    elif args.method == 'SERAC':
        hparams = SERACHparams

    print(f"Loading model... {args.model}")
    metrics = {}
    if args.no_edit:
        edited_model = AutoModelForCausalLM.from_pretrained(
            model_name, local_files_only=True, low_cpu_mem_usage=True)
        edited_model = edited_model.to(device)
    elif args.method == 'IKE':
        train_ds = [
            {
                'prompt': 'Q: The president of the US is? A:',
                'target_new': 'Joe Biden',
                'rephrase_prompt': 'The leader of the United State is',
                'locality_prompt': 'The president of Russia is ',
                'locality_ground_truth': 'Putin'
            },
            {
                'prompt': 'Einstein specialized in',
                'target_new': 'physics',
                'rephrase_prompt': 'Einstein is good at',
                'locality_prompt': 'Q: Which subject did Newton specialize in? A: ',
                'locality_ground_truth': 'physics'
            },
            # add more if needed
        ]
        hparams_config = hparams.from_hparams(config)
        editor = BaseEditor.from_hparams(hparams_config)
        # Initialize SentenceTransformer model
        sentence_model = SentenceTransformer(
            hparams_config.sentence_model_name)
        # Generate and save sentence embeddings
        encode_ike_facts(sentence_model, train_ds, hparams_config)
        metrics, edited_model, _ = editor.edit(
            prompts=prompts,
            ground_truth=ground_truth,
            rephrase_prompts=rephrase_prompts,  # new para
            target_new=target_news,
            locality_inputs=locality_inputs,
            subject=subjects,
            train_ds=train_ds,
            copy=True,
            return_orig_weights=True,
            keep_original_weight=True,
            test_generation=True,
            portability_inputs=portability_inputs
        )
    else:
        hparams_config = hparams.from_hparams(config)
        editor = BaseEditor.from_hparams(hparams_config)
        metrics, edited_model, _ = editor.edit(
            prompts=prompts,
            ground_truth=ground_truth,
            rephrase_prompts=rephrase_prompts,
            locality_inputs=locality_inputs,
            target_new=target_news,
            subject=subjects,
            test_generation=True,
            keep_original_weight=False,
            portability_inputs=portability_inputs
        )

    tokenizer_model = None
    if args.model == 'gpt2-xl':
        tokenizer_model = 'gpt2-xl'
    elif args.model == 'gptj':
        tokenizer_model = 'EleutherAI/gpt-j-6b'
    elif args.model == 'llama2-7b':
        tokenizer_model = 'meta-llama/Llama-2-7b-hf'
    elif args.model == 'llama2-7b-chat':
        tokenizer_model = 'meta-llama/Llama-2-7b-chat-hf'

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_model, local_files_only=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'

    subject_prompt = sample['coupled_prompts_and_properties']['subject_entity']['coupled_prompt'] + \
            f"\n-relationship to {sample['coupled_prompts_and_properties']['coupled_entities'][0]['entity']}"
    subject_prompt_guide = f"\n\n{subject}"

    coupled_prompt = sample['coupled_prompts_and_properties']['coupled_entities'][0]['coupled_prompt'] + \
            f"\n-relationship to {subject}"
    coupled_prompt_guide = f"\n\n{sample['coupled_prompts_and_properties']['coupled_entities'][0]['entity']}"

    if args.method == 'IKE':
        ike = f"Q: {prompt}? A: {target_new}\n"
        coupled_prompt = ike + coupled_prompt
        subject_prompt = ike + subject_prompt
        generation_prompts = [
            subject_prompt + subject_prompt_guide,
            coupled_prompt + coupled_prompt_guide,
        ]
    if args.model == 'llama2-chat':
        subject_prompt = subject_prompt + "[/INST]"
        coupled_prompt = coupled_prompt + "[/INST]"
        generation_prompts = [
            subject_prompt + subject_prompt_guide,
            coupled_prompt + coupled_prompt_guide,
        ]
    else:
        generation_prompts = [
            subject_prompt + subject_prompt_guide,
            coupled_prompt + coupled_prompt_guide,
        ]

    batch = tokenizer(
        generation_prompts,
        return_tensors='pt',
        padding=True
    )

    sampling_params = {}
    if args.use_sampling:
        sampling_params = {
            'do_sample': True,
            'top_k': 50,
            'top_p': 0.95,
            'temperature': 0.9,
            'num_return_sequences': 1
        }
    else:
        sampling_params = {
            'num_beams': 1,
            'early_stopping': True
        }
    
    generated_sample = sample
    for token_length in [600]:
        with torch.no_grad():
            post_edit_outputs = edited_model.generate(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                max_new_tokens=token_length,
                repetition_penalty=1.1,
                **sampling_params
            )
        generated_sample[f'subject_prompt_{token_length}'] = tokenizer.decode(
                post_edit_outputs[0].detach().cpu().numpy().tolist(), skip_special_tokens=True).replace(subject_prompt, '').strip(),
        generated_sample[f'coupled_prompt_{token_length}'] = tokenizer.decode(post_edit_outputs[1].detach().cpu().numpy().tolist(), skip_special_tokens=True).replace(coupled_prompt, '').strip(),
        

    return generated_sample, metrics or {}
