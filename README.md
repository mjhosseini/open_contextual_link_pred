UNDER DEVELOPMENT

<div class=figure>
  <p align="center"><img src="https://www.dropbox.com/s/ovxcb46pvk5bup2/toy_emnlp_findings2021.png?raw=1"
    width="450" height=auto></p>
  <p align="center"><small><i>a) Part of an example KG. b) The contexts from which we have extracted the KG triples. c) An example EG. The contextual link prediction and EG learning tasks are complementary. For example, acquire → own from the EG can independently be used to add the missing own relation to the KG.</i></small></p>
</div>

This codebase contains the implementation of the following paper:

**Open-Domain Contextual Link Prediction and its Complementarity with Entailment Graphs**, *Mohammad Javad Hosseini, Shay B. Cohen, Mark Johnson, and Mark Steedman. Findings of the Association for Computational Linguistics: EMNLP 2021.* [[paper]](https://aclanthology.org/2021.findings-emnlp.238.pdf)

# Setup

## Cloning the project and installing the requirements

    git clone https://github.com/mjhosseini/open_contextual_link_pred.git
    cd open_contextual_link_pred/
    sh scripts/requirements.sh

## Preparing the data

Download the extracted triple mentions and other necessary data into the folder "data/":
    
    sh scripts/data.sh

# Running the code

## Contextual link prediction

### Training

Training the CNCE (Contextualized and Non-Contextualized Embeddings) model on the NewsSpike corpus for the contextual link prediction task:

     python modeling/run_contextual_link_pred.py --model_type bert --model_name_or_path bert-base-uncased --do_train --do_lower_case --input_path data/news_bert_input.json --trels_folder data/typed_rels --all_triples_path data/NS_epair_split/ --learning_rate 5e-4 --ctx_lr_ratio 1e-2 --num_train_epochs 10 --max_seq_length 40 --output_dir models/CNCE_lr_5e-4_ctx_lr_ratio_1e-2_bsz_64_entity_pair_split --per_gpu_batch_size=64 --num_examples 8500000 --gradient_accumulation_steps 2 --overwrite_output --cache_dir . --logging_steps 20 --preferred_num_labels 0 --evaluate_during_training
    
In the above training, we make sure that the entity-pairs in the triple mentions in training, develpment, and test sets do not overlap. Therefore, the model cannot just memorize the relations that hold between entity pairs. The flag --all_triples_path provides a split of the data based on entity-pairs.

***See the meaning of all the flags in modeling/run_contextual_link_pred.py***

### Existing pre-trained models (optional)

Instead of training, you can download the pre-trained contextual link prediction models.

    sh scripts/dl_pretrained.sh

All the mentioned models in this GitHub page could be found in the folder 'pretrained_models'. The above model can be found in "pretrained_models/CNCE_lr_5e-4_ctx_lr_ratio_1e-2_bsz_64_entity_pair_split".

### Evaluation

Evaluating the CNCE model on the contextual link prediction task:

    python modeling/run_contextual_link_pred.py --model_type bert --model_name_or_path models/CNCE_lr_5e-4_ctx_lr_ratio_1e-2_bsz_64_entity_pair_split/checkpoint-631000  --do_test --do_lower_case --input_path data/news_bert_input.json --trels_folder data/typed_rels --all_triples_path data/NS_epair_split/ --max_seq_length 40 --per_gpu_batch_size=256 --cache_dir . --preferred_num_labels 100000

The results will be written in the following file: "test_final_CNCE_lr_5e-4_ctx_lr_ratio_1e-2_bsz_64_entity_pair_split.txt".

## Using contextual link prediction to improve entailment graph learning

## Training

Since the entailment graphs will be evaluated on a different dataset (e.g., Levy/Holt's dataset), we do not need the constraints on the entity pairs seen during training vs the ones during testing. We perform the training again without the --all_triples_path flag. In this case, the code splits the triple mentions randomly into training, development and test sets (with overlap between the entities). We observed that random split will yield slightly better results on entailment datasets because the training sees more diverse examples (almost all entity pairs occur in training).    

     python modeling/run_contextual_link_pred.py --model_type bert --model_name_or_path bert-base-uncased --do_train --do_lower_case --input_path data/news_bert_input.json --trels_folder data/typed_rels --learning_rate 5e-4 --ctx_lr_ratio 1e-2 --num_train_epochs 10 --max_seq_length 40 --output_dir models/CNCE_lr_5e-4_ctx_lr_ratio_1e-2_bsz_64_random_split --per_gpu_batch_size=64 --num_examples 8500000 --gradient_accumulation_steps 2 --overwrite_output --cache_dir . --logging_steps 20 --preferred_num_labels 0 --evaluate_during_training
    
You can also use the following model from the pre-trained folder: "pretrained_models/CNCE_lr_5e-4_ctx_lr_ratio_1e-2_bsz_64_random_split".

### Building entailment graphs

**CNCE MC**:

    python modeling/run_contextual_link_pred.py --model_type bert --model_name_or_path models/CNCE_lr_5e-4_ctx_lr_ratio_1e-2_bsz_64_random_split/checkpoint-631000 --do_build_entgraphs --do_lower_case --input_path data/news_bert_input.json --trels_folder data/typed_rels --max_seq_length 40 --num_fill 0 --per_gpu_batch_size=64 --entgraph_dir  entgraphs_CNCE_MC_random_split
    
The entailment graphs will be written in "entgraphs_CNCE_MC_random_split".
    
**CNCE AUG MC**:

    python modeling/run_contextual_link_pred.py --model_type bert --model_name_or_path models/CNCE_lr_5e-4_ctx_lr_ratio_1e-2_bsz_64_random_split/checkpoint-631000 --do_build_entgraphs --do_lower_case --input_path data/news_bert_input.json --trels_folder data/typed_rels --max_seq_length 40 --num_fill 100 --per_gpu_batch_size=512 --alpha=.5 --entgraph_dir entgraphs_AUG_CNCE_MC_random_split

The entailment graphs will be written in "entgraphs_AUG_CNCE_MC_random_split".

### Learning global entailment graphs (for Levy/Holt's dataset)
Please refer to https://github.com/mjhosseini/entGraph/ (step 6) for learning global entailment graphs from local entailment graphs (the ones that were learned above).

*We set the below parameters for the "CNCE MC" and "CNCE AUG MC" models:*

*constants.ConstantsGraphs*:

* root=".../entgraphs_CNCE_MC_random_split" (or ".../entgraphs_CNCE_AUG_MC_random_split")
* edgeThreshold=0.00005

*constants.ConstantsSoftConst*:

* lmbda=0.00005, lmbda_2=1.5, epsilon=.3, and tPropSuffix="_lpred_CNCE_MC_lmbda_0.00005_lmbda2_1.5_eps_.3" (or "_lpred_CNCE_AUG_MC_lmbda_0.00005_lmbda2_1.5_eps_.3").

### Evaluation on entailment datasets

See https://github.com/mjhosseini/entgraph_eval for the steps to evaluate the entailment graphs on the Levy/Holt's dataset.


## Using Entailment Graphs to Improve the Contextual link prediction task.

### Building entailment graphs

We build the entailment graphs again.

    python modeling/run_contextual_link_pred.py --model_type bert --model_name_or_path models/CNCE_lr_5e-4_ctx_lr_ratio_1e-2_bsz_64_entity_pair_split/checkpoint-631000 --do_build_entgraphs --do_lower_case --input_path data/news_bert_input.json --trels_folder data/typed_rels --all_triples_path data/NS_epair_split/ --use_only_training_data_to_build_entgraphs --max_seq_length 40 --num_fill 100 --per_gpu_batch_size=512 --alpha=.5 --entgraph_dir entgraphs_CNCE_AUG_MC_entity_pair_split_only_train

The above command has two differences with the previous command to build entailment graphs:

A. using entity-pairs splits as we are going to use the entailment graphs for the contextual link prediction task (so we cannot use the entailment graphs from the random triple mentions split).

B. We only use the training portion of the triple mentions by using the flag --use_only_training_data_to_build_entgraphs.

The entailment graphs will be written in "entgraphs_AUG_CNCE_MC_fill_100_bsz512_alpha_.5_entity_pair_split_only_train".

### Evaluating entailment graphs on the contextual link prediction task

In our preliminary experiments, we observed slightly better results for the local graphs compared to global graphs when used on the contextual link prediction task (while in our earlier work, global graphs did better than local graphs on the standard link prediction). Therefore, we used the local graphs to improve contextual link prediction.

    python modeling/run_contextual_link_pred.py --model_type bert --model_name_or_path models/CNCE_lr_5e-4_ctx_lr_ratio_1e-2_bsz_64_entity_pair_split/checkpoint-631000 --do_test --do_lower_case --input_path data/news_bert_input.json --trels_folder data/typed_rels/ --all_triples_path data/NS_epair_split/ --prebuilt_entgraph_dir entgraphs_CNCE_AUG_MC_entity_pair_split_only_train --max_seq_length 40 --per_gpu_batch_size=256 --cache_dir . --preferred_num_labels 100000

### Evaluating the combination of entailment graphs and the CNCE model on the contextual link prediction task
    
    python modeling/run_contextual_link_pred.py --model_type bert --model_name_or_path models/CNCE_lr_5e-4_ctx_lr_ratio_1e-2_bsz_64_entity_pair_split/checkpoint-631000  --do_test --do_lower_case --input_path data/news_bert_input.json --trels_folder data/typed_rels/ --all_triples_path data/NS_epair_split/ --do_eval_ext comb_beta_.9 --beta_comb .9 --prebuilt_entgraph_dir entgraphs_CNCE_AUG_MC_entity_pair_split_only_train --combine_entgraph_emb --max_seq_length 40 --per_gpu_batch_size=256 --cache_dir . --preferred_num_labels 100000


# Citation

If you found this codebase useful, please cite:

    @inproceedings{hosseini-etal-2021-open-domain,
    title = "Open-Domain Contextual Link Prediction and its Complementarity with Entailment Graphs",
    author = "Hosseini, Mohammad Javad  and
      Cohen, Shay B.  and
      Johnson, Mark  and
      Steedman, Mark",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    year = "2021",
    pages = "2790--2802",
    }


