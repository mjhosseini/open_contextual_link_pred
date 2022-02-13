UNDER DEVELOPMENT

This codbase contains the implementation of the following paper:

**Open-Domain Contextual Link Prediction and its Complementarity with Entailment Graphs**, *Mohammad Javad Hosseini, Shay B. Cohen, Mark Johnson, and Mark Steedman. Findings of the Association for Computational Linguistics: EMNLP 2021.* [[paper]](https://aclanthology.org/2021.findings-emnlp.238.pdf)

## Setup

### Cloning the project and installing the requirements

    git clone https://github.com/mjhosseini/open_contextual_link_pred.git
    cd open_contextual_link_pred/
    sh scripts/requirements.sh

### Preparing the data

Download the extracted binary relations from the NewsSpike corpus into convE/data folder:
    
    sh scripts/data.sh

## Running the code

### Contextual link prediction

#### Training

Training the CNCE (Contextualized and Non-Contextualized Embeddings) model for the contextual link prediction task:

     python modeling/run_contextual_link_pred.py --model_type bert --model_name_or_path bert-base-uncased --do_train --do_lower_case --input_path ../contextual_data/news_bert_input.json --trels_folder ../contextual_data/typed_rels --all_triples_path ../contextual_data/NS_epair_split/ --learning_rate 5e-4 --ctx_lr_ratio 1e-2 --num_train_epochs 10 --max_seq_length 40 --output_dir models_bert/contextual_5e-4_1e-2_bsz_64_epair_split --per_gpu_batch_size=64 --num_examples 8500000 --gradient_accumulation_steps 2 --overwrite_output --cache_dir . --logging_steps 20 --preferred_num_labels 0 --evaluate_during_training
    
Alternatively, you can copy the pre-trained contextual link prediction model (TODO).

    sh scripts/dl_pretrained.sh

#### Evaluation

Evaluating the CNCE model on the contextual link prediction task:



### Computing triple (link) probabilities for seen and unseen triples

**Only on training triples:**

    CUDA_VISIBLE_DEVICES=0 python3 convE/main.py model ConvE input_drop 0.2 hidden_drop 0.3 feat_drop 0.2 lr 0.003 lr_decay 0.995 dataset NS_probs_train process True mode probs probs_file_path NS_probs_train.txt

**On all triples:**

    CUDA_VISIBLE_DEVICES=0 python3 convE/main.py model ConvE input_drop 0.2 hidden_drop 0.3 feat_drop 0.2 lr 0.003 lr_decay 0.995 dataset NS_probs_all process True mode probs probs_file_path NS_probs_all.txt

### Building the entailment graphs

Build the entailment graphs by the Marcov Chain model (random walk) as well as the Marcov Chain model (random walk) + augmentation with new scores. The former is done by --max_new_args 0 and the latter is done by --max_new_args 50. 

This step should be run on CPU, preferably with more than 100GB RAM (depending on the --max_new_args parameter).

**Only for training triples:**

    python randWalk/randWalkMatFactory.py --probs_file_path NS_probs_train.txt --triples_path convE/data/NS/train.txt --max_new_args 0 --entgraph_path typedEntGrDir_NS_train_MC
    python randWalk/randWalkMatFactory.py --probs_file_path NS_probs_train.txt --triples_path convE/data/NS/train.txt --max_new_args 50 --entgraph_path typedEntGrDir_NS_train_AUG_MC

**On all triples:**

    python randWalk/randWalkMatFactory.py --probs_file_path NS_probs_all.txt --triples_path convE/data/NS/all.txt --max_new_args 0 --entgraph_path typedEntGrDir_NS_all_MC
    python randWalk/randWalkMatFactory.py --probs_file_path NS_probs_all.txt --triples_path convE/data/NS/all.txt --max_new_args 50 --entgraph_path typedEntGrDir_NS_all_AUG_MC

## Evaluation

### Evaluate the entailment graphs

Please refer to https://github.com/mjhosseini/entgraph_eval for evaluation.

We can use the entailment graphs that are learned by accessing all the link prediction data as here we only evaluate the entailment task, not link prediction task. Use the learned entailment graphs (typedEntGrDir_NS_all_MC or typedEntGrDir_NS_all_AUG_MC) as the gpath parameter of the entgraph_eval project.

### Improve link prediction with entailment graphs

We can use the entailment graphs that are learned by accessing only the link prediciton training data.

Using entailment graphs with the Marcov Chain model (random walk):

    CUDA_VISIBLE_DEVICES=0 python3 convE/main.py model ConvE input_drop 0.2 hidden_drop 0.3 feat_drop 0.2 lr 0.003 lr_decay 0.995 dataset NS process True mode test_entgraphs entgraph_path typedEntGrDir_NS_train_MC 1>lpred_detailed_output_MC.txt 2>&1 &

Using entailment graphs with the Marcov Chain model (random walk) + augmentation with new scores:

    CUDA_VISIBLE_DEVICES=0 python3 convE/main.py model ConvE input_drop 0.2 hidden_drop 0.3 feat_drop 0.2 lr 0.003 lr_decay 0.995 dataset NS process True mode test_entgraphs entgraph_path typedEntGrDir_NS_all_MC 1>lpred_detailed_output_MC.txt 2>&1 &

## Citation

If you found this codebase useful, please cite:

    @inproceedings{hosseini2019duality,
      title={Duality of Link Prediction and Entailment Graph Induction},
      author={Hosseini, Mohammad Javad and Cohen, Shay B and Johnson, Mark and Steedman, Mark},
      booktitle={Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
      pages={4736--4746},
      year={2019}
    }

