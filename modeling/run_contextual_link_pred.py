# coding=utf-8
# This code is adapted from Huggingface transformer's codebase
# (https://github.com/huggingface/transformers/blob/master/examples/pytorch/multiple-choice/run_swag.py) for the
# contextual link prediction and entailment graph learning tasks.
""" Finetuning the library models for contextual link prediction (Bert) and using the results to build entailment graphs."""

import argparse
import glob
import logging
import os
import random
import sys

sys.path.append("..")
sys.path.append(".")

import numpy as np
import torch

from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from collections import defaultdict

from transformers import (
    AdamW,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

from modeling_bert_contextual_link_pred import BertForEntGraphs, BertConfig

from utils_contextual_link_pred import News_Iterable_Dataset, EntGraphAggregator, PreBuiltEntGraphCollection

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForEntGraphs, BertTokenizer),
}


def _simple_accuracy(preds, labels):
    return (preds == labels).mean()


def _set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def evaluate(args, model, iter_eval_dataloader, eval_dataset, eval_file, prefix="", prebuilt_entgr_col=None,
             reset=False, tokenizer=None):
    # multi-gpu evaluate
    if args.n_gpu > 1 and model:
        model = torch.nn.DataParallel(model)

    # Computing Entailment Scores!
    logger.info("***** Evaluating Entailmet Scores {} *****".format(prefix))
    # logger.info("  Num examples = %d", len(all_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0

    preds = None
    out_labels = None
    batch_orig = None

    # for step, batch_orig in enumerate(tqdm_obj):
    for step in range(args.num_batches_eval):
        try:
            batch_orig = next(iter_eval_dataloader)
        except StopIteration:
            if eval_dataset.finished and reset:
                print("RESETTING evaluate_dataset and eval_dataloader")
                eval_dataset = get_iterable_dataset(args, eval_dataset.entgrAgg, evaluate=True, tokenizer=tokenizer)
                iter_eval_dataloader = iter(DataLoader(eval_dataset, batch_size=args.eval_batch_size))
                batch_orig = next(iter_eval_dataloader)
            else:
                batch_orig = None
                break

        batch_orig = tuple(t for t in batch_orig.values())

        ent_scores_model = ent_scores_entgraph = None

        if model:
            model.eval()
            with torch.no_grad():
                batch = tuple(t.to(args.device) for t in batch_orig)
                inputs = {
                    "input_ids": batch[0],
                    "attention_masks": batch[1],
                    "pred_cntx_idxes": batch[3],
                    "labels": batch[4],
                    "label_idxes": batch[5],
                    "pred_cntx_idxes_end": batch[8],
                    "linear_masks": batch[9]
                }

                if args.contextual_hyp:
                    inputs.update({
                        "hyp_ids": batch[10],
                        "hyp_attention_masks": batch[11],
                        "hyp_start_idxes": batch[12],
                        "hyp_end_idxes": batch[13]
                    })

                outputs = model(**inputs)
                ent_scores_model = outputs[-1]
                ent_scores_model = ent_scores_model.detach().cpu().numpy()

                tmp_eval_loss = outputs[0]
                eval_loss += tmp_eval_loss.mean().item()
        if prebuilt_entgr_col:  # evaluating with pre-built entailment graph collection
            ent_scores_entgraph = prebuilt_entgr_col.get_ent_scores(batch_orig)

        if model and prebuilt_entgr_col:
            ent_scores = ent_scores_model * (1 - args.beta_comb) + args.beta_comb * ent_scores_entgraph
        elif model:
            ent_scores = ent_scores_model
        else:
            ent_scores = ent_scores_entgraph

        nb_eval_steps += 1

        print("current batch size: " + str(eval_dataset.current_batch_size))

        ent_scores = ent_scores[:eval_dataset.current_batch_size]

        labels = batch_orig[4].numpy()[:eval_dataset.current_batch_size]

        eval_dataset.entgrAgg.set_self_score_one(ent_scores, batch_orig)

        if preds is None:
            preds = ent_scores
            out_labels = labels
        else:
            preds = np.append(preds, ent_scores, axis=0)
            out_labels = np.append(out_labels, labels, axis=0)

        print("eval step: " + str(step))

    if batch_orig:
        if args.write_output:
            eval_dataset.entgrAgg.report_eval_scores(batch_orig, preds, eval_dataset.current_batch_size)
        eval_loss = eval_loss / nb_eval_steps
        AP = average_precision_score(out_labels, preds, average="samples")
        preds[preds >= .5] = 1.0
        preds[preds < .5] = 0.0
        acc = _simple_accuracy(preds, out_labels)
        results = {"eval_acc": acc, "eval_loss": eval_loss, "eval_AP": AP}
        print("writing to eval_file")
        eval_file.write(str(AP) + "\t" + str(eval_loss) + "\t" + str(
            (eval_dataset.current_batch_size) / eval_dataset.batch_size) + "\n")
        eval_file.flush()
    else:
        results = None

    return results, iter_eval_dataloader, eval_dataset


def build_entgraphs(args, all_dataset, model, entgrAgg, prefix=""):
    if args.entscore_mode == "contextual":
        args.entgraph_batch_size = args.per_gpu_batch_size * max(1, args.n_gpu)
    else:
        args.entgraph_batch_size = args.per_gpu_batch_size

    all_dataloader = DataLoader(all_dataset, batch_size=args.entgraph_batch_size)

    print("len(all_dataloader)", len(all_dataloader))
    print("len(all_dataset)", len(all_dataloader))

    # multi-gpu evaluate
    if args.n_gpu > 1 and model:
        model = torch.nn.DataParallel(model)

    # Computing Entailment Scores!
    logger.info("***** Building Entailment Graphs {} *****".format(prefix))
    # logger.info("  Num examples = %d", len(all_dataset))
    logger.info("  Batch size = %d", args.entgraph_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0

    for step, batch_orig in enumerate(tqdm(all_dataloader, desc="Ent Graph Building")):
        if model:
            model.eval()
        batch_orig = tuple(t for t in batch_orig.values())

        # if step == 1000:
        #     break

        if args.entscore_mode == "contextual":

            with torch.no_grad():
                batch = tuple(t.to(args.device) for t in batch_orig)
                inputs = {
                    "input_ids": batch[0],
                    "attention_masks": batch[1],
                    "pred_cntx_idxes": batch[3],
                    "labels": None,
                    "label_idxes": batch[5],
                    "pred_cntx_idxes_end": batch[8],
                    "linear_masks": batch[9]
                }

                outputs = model(**inputs)
                ent_scores = outputs[-1].detach().cpu()

        elif args.entscore_mode == "binary":
            ent_scores = batch_orig[4]  # labels!
        else:
            raise Exception("not implemented!")

        entgrAgg.update_entgraph(batch_orig, ent_scores)

        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps

    print("eval_loss average: ", eval_loss)

    entgrAgg.write_graphs(args.entgraph_dir)


def train(args, train_dataset, eval_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_batch_size * max(1, args.n_gpu)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size)

    eval_file = open("eval_" + args.output_dir.split("/")[1] + ".txt", "w")

    iter_eval_dataloader = None
    if args.evaluate_during_training:
        args.eval_batch_size = args.per_gpu_batch_size * max(1, args.n_gpu)
        eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size)
        iter_eval_dataloader = iter(eval_dataloader)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    print("t_total: ", t_total)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       not any(nd in n for nd in no_decay) and not "encoder.layer" in n],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       not any(nd in n for nd in no_decay) and "encoder.layer" in n],
            "weight_decay": args.weight_decay,
            "lr": args.learning_rate * args.ctx_lr_ratio
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    best_dev_acc = 0.0
    best_steps = 0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    _set_seed(args)  # Added here for reproductibility
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch.values())
            inputs = {
                "input_ids": batch[0],
                "attention_masks": batch[1],
                "pred_cntx_idxes": batch[3],
                "labels": batch[4],
                "label_idxes": batch[5],
                "pred_cntx_idxes_end": batch[8],
                "linear_masks": batch[9]
            }

            if args.contextual_hyp:
                inputs.update({
                    "hyp_ids": batch[10],
                    "hyp_attention_masks": batch[11],
                    "hyp_start_idxes": batch[12],
                    "hyp_end_idxes": batch[13]
                })

            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in contextual_link_pred (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            if step % 10 == 0:
                print("loss: ", loss)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if (
                            args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results, iter_eval_dataloader, eval_dataset = evaluate(args, model, iter_eval_dataloader,
                                                                               eval_dataset, eval_file, reset=True)
                        for key, value in results.items():
                            print("eval_{}".format(key), value, global_step)

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_vocabulary(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if step % 100 == 0:
                print("learning rates:")
                for param_group in optimizer.param_groups:
                    print(param_group["lr"])

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

        print("EPOCH COMPLETED" + str(epoch))

        if epoch < args.num_train_epochs:
            print("RESETTING train_dataset and train_dataloader")
            train_dataset = get_iterable_dataset(args, train_dataset.entgrAgg, evaluate=False, tokenizer=tokenizer)
            train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size)

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step, best_steps


def get_iterable_dataset(args, entgrAgg, all=False, evaluate=False, test=False, tokenizer=None):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Load data features from cache or dataset file
    if all:
        data_mode = "all"
    elif evaluate:
        data_mode = "dev"
    elif test:
        data_mode = "test"
    else:
        data_mode = "train"
    assert not (evaluate and test)
    input_path = args.input_path

    if args.entscore_mode == "contextual":
        args.batch_size = args.per_gpu_batch_size * max(1, args.n_gpu)
    else:
        args.batch_size = args.per_gpu_batch_size
    dataset = News_Iterable_Dataset(
        input_path, entgrAgg, data_mode, args.batch_size, args.preferred_num_labels, args.neg_ratio,
        args.max_seq_length, args.num_examples, args.mask_ents_prob, tokenizer, args.contextual_hyp, args.hard_negs,
        args.process_idx, args.num_processes, args.updated_len_loader, args.no_argord, args.triple_count_path)
    return dataset


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--input_path",
        default=None,
        type=str,
        required=True,
        help="The input data path. It should contain the json file containing the parsed triple mentions and their"
             "contexts.",
    )

    parser.add_argument(
        "--trels_folder",
        default=None,
        type=str,
        required=True,
        help="The input trels folder: For each type-pair, the predicates for each timestamp#entity_1#entity_2 are "
             "recorded. For each timestamp and entity pair, the predicates in an interval around the timestamp are "
             "considered. In all our experiments, the interval is set to infinity, so the timestamps are basically not "
             "used. Preliminary experiments with smaller intervals didn't show any improvements in entailment graph "
             "learning. It is possible to use the code with smaller intervals if we re-generate the trels folder."
    )

    parser.add_argument(
        "--all_triples_path",
        default=None,
        type=str,
        required=False,
        help="The input triples split folder (train, dev, and test). If supplied, we split the tripe mentions based on"
             "those triples. Otherwise, we split the triple mentions randomly."
    )

    parser.add_argument(
        "--num_batches_eval",
        type=int,
        default=1,
        help="Deprecated. It should be 1. previously: number of batches to evaluate. If -1, evaluate everyting!",
    )

    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name",
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters

    parser.add_argument(
        "--tokenizer_name",  # Not important
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )

    parser.add_argument(
        "--max_seq_length",
        default=40,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument(
        "--pred_dims",
        default=-1,
        type=int,
        help="The number of dimensions for the predicate. By default (-1), it will be set to hidden_size",
    )

    parser.add_argument(
        "--num_examples",
        default=8500000,
        type=int,
        help="The number of examples (triple mentions) in the corpus. Default is for the NewsSpike corpus.",
    )

    parser.add_argument(
        "--preferred_num_labels",
        type=int,
        default=0,
        help="The number of (extra) hypothesis predicates that we want to compute contextual link prediction scores for."
             " We always compute the contextual link prediction scores for all the predicates in the current batch. We "
             "either set this to A) 0, that means we don't evaluate any extra predicates and the candidate predicates"
             "will be only the ones in the current batch (used for training and building entailment graphs for"
             "performance reasons); or B)infinity (100000 for the NewsSpike corpus) which means that we want the"
             "candidate predicates to contain all the possible predicates in the corpus (used for evaluation of the"
             "contextual link prediction)."
    )

    parser.add_argument(
        "--num_fill",
        default=100,
        type=int,
        help="When building the entailment graphs using the CNCE Markov Chain model, if the number of observed "
             "connected predicates for a triple mention is less than num_fill, we augment the Markov Chain by "
             "connecting the mentions to predicted predicates so that the number of total connections becomes num_fill "
             "(the parameter K in the paper). If the number of all predicates with the same types is less than "
             "num_fill, the triple mention will be connected to all existing predicates with the same types."
    )

    parser.add_argument(
        "--neg_ratio",
        type=float,
        default=1.0,
        help="Number of negative labels per positive. Can be used in training, but not for eval or building the graphs."
             "Was not used in the paper experiments as it did not yield improved results.",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=.5,
        help="The combination coefficient for the novel connection (the parameter alpha in the paper). Used when "
             "building entailment graphs",
    )

    parser.add_argument(
        "--beta_comb",
        type=float,
        default=.5,
        help="This is the beta parameter in the paper, i.e., the weight to put on entailment graphs (1-beta will be "
             "put on the contextual model). Used when doing contextual link prediction based on the combination of the"
             "trained model and the entailment graphs.",
    )

    parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help="number of processes for building entailment graphs. Was 1 for the NewsSpike experiments.",
    )

    parser.add_argument(
        "--process_idx",
        type=int,
        default=0,
        help="process_idx for building entailment graphs. Useful when num_processes is higher than 1.",
    )

    parser.add_argument(
        "--entgraph_dir",
        default="entgraphs",
        type=str,
        help="The output entgraph_dir, used with do_build_entgraph",
    )

    parser.add_argument(
        "--embs_init_path",
        default=None,
        type=str,
        help="The pred_embs_init path for either reading or writing. Not used in the current experiments",
    )

    parser.add_argument(
        "--do_eval_ext",
        default=None,
        type=str,
        help="eval_file extension just for easier distinction between output files.",
    )

    parser.add_argument(
        "--entscore_mode",
        default="contextual",
        type=str,
        help="Entailment graph score mode: contextual, binary, or count. All the paper experiments were done with the"
             "`contextual` mode",
    )

    parser.add_argument(
        "--device_name",
        default=None,
        type=str,
        help="device_name used in torch.",
    )

    parser.add_argument(
        "--prebuilt_entgraph_dir",
        default=None,
        type=str,
        help="The pre-built entailment graph directory on only the training portion of the triples. Used when evaluating"
             "the entailment graphs (either alone or combined with the contextual model) on the contextual link"
             "prediction task",
    )

    parser.add_argument(
        "--prebuilt_simSuffix",
        default="_sim.txt",
        type=str,
        help="Prebuilt Entailment graph simSuffix, i.e., the suffix at the end of the entailment graph files (e.g.,"
             "person#location_sim.txt)",
    )

    parser.add_argument(
        "--prebuilt_featIdx",
        default=0,
        type=int,
        help="Prebuilt Entailment graph featIdx. 0 means the first feature (similarity score) in the entailment graph "
             "files.",
    )

    parser.add_argument(
        "--triple_count_path",
        default=None,
        type=str,
        help="Where to write triple_count_path. Will be used in evaluating normal link pred models.",
    )

    parser.add_argument("--mask_ents_prob", default=0.0, type=float,
                        help="Probability of masking entities with types during training. Not used in the paper")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_build_entgraphs", action="store_true", help="Whether to run build entailment graphs.")
    parser.add_argument("--use_only_training_data_to_build_entgraphs",
                        action="store_true",
                        help="Whether the entailment graphs are built using only the training portion of the triple "
                             "mentions, or the full triple mentions (training, dev, and test sets).")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action="store_true", help="Whether to run test on the test set")
    parser.add_argument("--combine_entgraph_emb", action="store_true",
                        help="Whether to combine ent graphs and emb model"
                             " for contextual link prediction.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )
    parser.add_argument("--freeze11", action="store_true", help="Whether to freeze the first 11 layers of BERT")
    parser.add_argument("--freeze12", action="store_true", help="Whether to freeze the first 12 layers of BERT. Used "
                                                                "for ablation studies.")
    parser.add_argument("--no_argord", action="store_true",
                        help="Whether to not do arg order mapping. Used for ablation"
                             " studies")
    parser.add_argument("--updated_len_loader", action="store_true", help="Whether to update len function to not divide"
                                                                          "by bsz")
    parser.add_argument("--write_output", action="store_true", help="Whether to write output of inference for ent graph"
                                                                    "building or evaluation")

    parser.add_argument("--contextual_hyp", action="store_true", help="Contextual hyp or not. Not used in the paper.")

    parser.add_argument("--pmi", action="store_true", help="Whether to weight scores by PMI between entpair and pred."
                                                           "Not used in the paper")

    parser.add_argument("--hard_negs", action="store_true",
                        help="Whether to generate hard negative examples instead of "
                             "random ones. Not used in the paper")

    parser.add_argument("--per_gpu_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--ctx_lr_ratio",
                        default=1.0,
                        type=float,
                        help="The initial learning rate ratio for contextualized embeddings vs other parameters. The"
                             "learning rate for contextualized embeddings is: learning_rate * ctx_lr_ratio")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=5000, help="Save checkpoint every X updates steps.")

    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()

    if not args.do_train:
        args.output_dir = args.model_name_or_path

    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    if args.pred_dims != -1:
        print("args.pred_dims: " + str(args.pred_dims))
        from modeling_bert_contextual_link_pred import BertForEntGraphsHiddenSize as BertForEntGraphs
        MODEL_CLASSES["bert"] = (BertConfig, BertForEntGraphs, BertTokenizer)

    if args.num_processes > 1 and not args.do_build_entgraphs:
        raise ValueError("Multi processing only possible when building the graphs.")

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        if not args.device_name:
            device = torch.device(
                "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")  # TODO (fix this)
        else:
            device = torch.device(
                args.device_name if torch.cuda.is_available() and not args.no_cuda else "cpu")  # TODO (fix this)
        if args.evaluate_during_training:  # TODO (change this, only n_gpu=1 for evaluation)
            print("setting n_gpu to 1")
            args.n_gpu = 1
        else:
            args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
            if args.evaluate_during_training:  # TODO (remove this)
                args.n_gpu = 1
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    _set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    entgrAgg = EntGraphAggregator(args.trels_folder, args.pmi, args.hard_negs, -1, args.all_triples_path, args.num_fill,
                                  args.alpha, args.write_output)

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    config = config_class.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    config.num_all_labels = int(entgrAgg.num_all_labels)
    config.pred_dims = args.pred_dims

    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    # don't load model if building entgraph with binary mode
    loadModel = not ((args.do_build_entgraphs and args.entscore_mode == "binary") or (
                (args.do_eval or args.do_test) and args.prebuilt_entgraph_dir and not args.combine_entgraph_emb))

    if loadModel:

        print("loading pretrained model")
        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        print("model loaded")
        #

        # TODO: remove below
        # model = model_class(config)
        # model.eval()
        # model.test_pretrained()

        if args.freeze11:
            unfrozen_layers = ["pooler", "cls", "pred_embs", "encoder.layer.11"]
            for name, param in model.named_parameters():
                if not any([layer in name for layer in unfrozen_layers]):
                    print("[FROZE]: %s" % name)
                    param.requires_grad = False
                else:
                    print("[FREE]: %s" % name)
                    param.requires_grad = True

        if args.freeze12:
            unfrozen_layers = ["pooler", "cls", "pred_embs"]
            for name, param in model.named_parameters():
                if not any([layer in name for layer in unfrozen_layers]):
                    print("[FROZE]: %s" % name)
                    param.requires_grad = False
                else:
                    print("[FREE]: %s" % name)
                    param.requires_grad = True

    else:
        model = None

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    if model:
        model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)
    best_steps = 0

    # Training
    if args.do_train:
        args.num_batches_eval = 1
        train_dataset = get_iterable_dataset(args, entgrAgg, evaluate=False, tokenizer=tokenizer)
        eval_dataset = None
        if args.evaluate_during_training:
            eval_dataset = get_iterable_dataset(args, entgrAgg, evaluate=True, tokenizer=tokenizer)

        if args.embs_init_path:
            embs = entgrAgg.get_init_emb_weights(args.embs_init_path)
            model.init_emb_weights(embs)
        global_step, tr_loss, best_steps = train(args, train_dataset, eval_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Building entailment graphs
    if args.do_build_entgraphs:

        if not args.use_only_training_data_to_build_entgraphs:
            all_dataset = get_iterable_dataset(args, entgrAgg, all=True, tokenizer=tokenizer)
        else:
            all_dataset = get_iterable_dataset(args, entgrAgg, evaluate=False, tokenizer=tokenizer)
        checkpoints = [args.output_dir]
        logger.info("Building the entailment graphs with the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
            model = None
            if loadModel:
                print("loading model for build ent graph")
                model = model_class.from_pretrained(checkpoint)
                model.to(args.device)
            build_entgraphs(args, all_dataset, model, entgrAgg, prefix)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    if (args.do_eval or args.do_test) and args.local_rank in [-1, 0]:
        # if not args.do_train:
        #     args.output_dir = args.model_name_or_path

        args.num_batches_eval = 1

        args.eval_batch_size = args.per_gpu_batch_size * max(1, args.n_gpu)
        if args.do_eval:
            eval_dataset = get_iterable_dataset(args, entgrAgg, evaluate=True, tokenizer=tokenizer)
        else:
            eval_dataset = get_iterable_dataset(args, entgrAgg, test=True, tokenizer=tokenizer)

        eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size)

        model = None
        prebuilt_entgr_col = None

        global_step = 0

        if loadModel:

            checkpoints = [args.output_dir]
            logger.info("Evaluate the following checkpoints: %s", checkpoints)
            for checkpoint in checkpoints:
                print("loading model for evaluation")
                model = model_class.from_pretrained(checkpoint)
                model.to(args.device)

        if args.prebuilt_entgraph_dir:
            prebuilt_entgr_col = PreBuiltEntGraphCollection(args.prebuilt_entgraph_dir, args.prebuilt_featIdx,
                                                            args.prebuilt_simSuffix, entgrAgg)

        if args.combine_entgraph_emb:
            if args.do_eval:
                eval_file = open("eval_final_all_" + args.model_name_or_path.split("/")[1] + "_" +
                                 args.prebuilt_entgraph_dir.split("/")[-1] + str(args.prebuilt_featIdx) + (
                                     "_" + args.do_eval_ext if args.do_eval_ext else "") + ".txt", "w")
            else:
                eval_file = open("test_final_all_" + args.model_name_or_path.split("/")[1] + "_" +
                                 args.prebuilt_entgraph_dir.split("/")[-1] + str(args.prebuilt_featIdx) + (
                                     "_" + args.do_eval_ext if args.do_eval_ext else "") + ".txt", "w")

        elif loadModel:
            if args.do_eval:
                eval_file = open("eval_final_all_" + args.model_name_or_path.split("/")[1] + (
                    "_" + args.do_eval_ext if args.do_eval_ext else "") + ".txt", "w")
            else:
                eval_file = open("test_final_all_" + args.model_name_or_path.split("/")[1] + (
                    "_" + args.do_eval_ext if args.do_eval_ext else "") + ".txt", "w")
        else:
            if args.do_eval:
                eval_file = open(
                    "eval_final_all_" + args.prebuilt_entgraph_dir.split("/")[-1] + str(args.prebuilt_featIdx) + (
                        "_" + args.do_eval_ext if args.do_eval_ext else "") + ".txt", "w")
            else:
                eval_file = open(
                    "test_final_all_" + args.prebuilt_entgraph_dir.split("/")[-1] + str(args.prebuilt_featIdx) + (
                        "_" + args.do_eval_ext if args.do_eval_ext else "") + ".txt", "w")

        iter_eval_dataloader = iter(eval_dataloader)

        eval_AP = 0
        eval_loss = 0

        types2AP = defaultdict(float)
        types2loss = defaultdict(float)
        types2count = defaultdict(float)

        while not eval_dataset.finished:
            results, iter_eval_dataloader, eval_dataset = evaluate(args, model, iter_eval_dataloader, eval_dataset,
                                                                   eval_file, prebuilt_entgr_col=prebuilt_entgr_col,
                                                                   tokenizer=tokenizer)
            current_batch_size, current_batch_type = eval_dataset.current_batch_size, eval_dataset.current_batch_types
            current_ratio = current_batch_size / eval_dataset.batch_size
            global_step += current_ratio
            types2count[eval_dataset.current_batch_types] += current_ratio
            if results:
                for key, value in results.items():
                    print("eval_{}".format(key), value, current_batch_size)
                    if "_AP" in key:
                        eval_AP += value * current_ratio
                        types2AP[eval_dataset.current_batch_types] += value * current_ratio
                    elif "_loss" in key:
                        eval_loss += value * current_ratio
                        types2loss[eval_dataset.current_batch_types] += value * current_ratio

        eval_file.write("\ntypes results:\n\n")

        types_results = []
        for types in types2count:
            types_results.append((types2AP[types] / types2count[types], types2loss[types] / types2count[types], types,
                                  types2count[types]))

        types_results = sorted(types_results, key=lambda x: -x[3])
        for types_result in types_results:
            for res in types_result:
                eval_file.write(str(res) + "\t")
            eval_file.write("\n")

        eval_file.write("\naccumulated results:\n\n")

        eval_MAP = eval_AP / global_step
        eval_file.write("MAP: " + str(eval_MAP) + "\n")

        eval_Mloss = eval_loss / global_step
        eval_file.write("mean eval_loss: " + str(eval_Mloss) + "\n")

    if best_steps:
        logger.info("best steps of eval acc is the following checkpoints: %s", best_steps)
    return results


if __name__ == "__main__":
    main()
