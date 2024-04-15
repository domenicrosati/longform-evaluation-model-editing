# longform-evaluation-model-editing
The datasets, models, and demonstration code for [long-form evaluation of model editing](https://arxiv.org/abs/2402.09394)

## Evaluating Longform Model Edit outputs with Annotation and Survey LLMs

Please see the following files to understand how to run the annotation and survey LLMs on the model edit outputs:
- `llm_annotator.py` for the annotation LLM techniques
- `llm_survey.py` for the survey LLM techniques

## Datasets

The datasets are available in the `data` directory:
- `data/counterfact_with_coupled_entities.json`: The Counterfact dataset with coupled entities
- `data/zsre_mend_eval_with_coupled_entities.json`: The ZSRE-MEND evaluation dataset with coupled entities

The paper uses samples from these datasets to evaluate the performance of model editing techniques.

We also provide the survey results and annotation results in the `results` directory.

## Code for experiments

The code is available in the `src` directory and provides the main functionality for the experiments in the apper.

The code outside of the `src` directory is used to demonstrate the experiments run in the paper.

## Citation

Cite using:
```
@misc{rosati2024longform,
      title={Long-form evaluation of model editing}, 
      author={Domenic Rosati and Robie Gonzales and Jinkun Chen and Xuemin Yu and Melis Erkan and Yahya Kayani and Satya Deepika Chavatapalli and Frank Rudzicz and Hassan Sajjad},
      year={2024},
      eprint={2402.09394},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```