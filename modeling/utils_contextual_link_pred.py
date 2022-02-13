# coding=utf-8
""" Contextual Link prediction fine-tuning and entailment graph learning: utilities used in run_contextual_link_pred.py."""

import json
import logging
import os
import random
import torch
import numpy as np
import subprocess
import argparse
from graph import Graph
import traceback

from torch.utils.data import IterableDataset
from collections import defaultdict
from copy import deepcopy


logger = logging.getLogger(__name__)

_ADD_TOKENS = True # used for logging purposes.


class PredInCntxExample:
    def __init__(self, input_ids, gr_idx, pred_idx_in_gr, entPairId, ent1, ent2, pred_cntx_idx, pred_cntx_idx_end, pos_labels, max_seq_len, pred_reversed, tokens=None, triple=None):
        l_idx = max(0, pred_cntx_idx - max_seq_len // 2)
        r_idx = min(len(input_ids), pred_cntx_idx + max_seq_len // 2)
        pred_cntx_idx_end = min(pred_cntx_idx_end, r_idx-1)
        self.input_ids = input_ids[l_idx:r_idx]
        #We make sure to start and end with [CLS] and [SEP]

        self.input_ids[0] = input_ids[0]
        self.input_ids[-1] = input_ids[-1]
        if _ADD_TOKENS:
            self.tokens = tokens[l_idx:r_idx]
            self.tokens[0] = tokens[0]
            self.tokens[-1] = tokens[-1]
            self.triple = triple
        self.attention_mask = [1] * len(self.input_ids)
        self.gr_idx = gr_idx
        self.pred_idx_in_gr = pred_idx_in_gr
        self.entPairId = entPairId
        self.ent1 = ent1
        self.ent2 = ent2
        self.pred_cntx_idx = pred_cntx_idx - l_idx
        self.pred_cntx_idx_end = pred_cntx_idx_end - l_idx
        self.pos_labels = pos_labels
        self.pred_reversed = pred_reversed
        self.labels = None
        self.label_idxes = None

        self.hyp_ids = None
        self.hyp_attention_masks = None
        self.hyp_start_idxes = None
        self.hyp_end_idxes = None

    def set_labels(self, all_label_idxes, label_offset):
        self.label_idxes = [label_idx + label_offset for label_idx in all_label_idxes]
        self.labels = [0] * len(all_label_idxes)
        pos_labels_set = set(self.pos_labels)
        for idx, label in enumerate(all_label_idxes):
            if label in pos_labels_set:
                self.labels[idx] = 1

    # Not used in the paper experiments.
    def set_hyp_tokeniztion_info(self, hyp_ids, hyp_attention_masks, hyp_start_idxes, hyp_end_idxes):
        self.hyp_ids = hyp_ids
        self.hyp_attention_masks = hyp_attention_masks
        self.hyp_start_idxes = hyp_start_idxes
        self.hyp_end_idxes = hyp_end_idxes

    def pad(self, max_len):
        # Zero-pad up to the sequence length.
        padding_length = max_len - len(self.input_ids)
        pad_token = 0
        self.input_ids = self.input_ids + ([pad_token] * padding_length)
        self.attention_mask = self.attention_mask + ([0] * padding_length)

    def get_tensor_dict(self):

        input_ids = torch.tensor(self.input_ids, dtype=torch.long)
        attention_mask = torch.tensor(self.attention_mask, dtype=torch.long)
        gr_idx = torch.tensor(self.gr_idx, dtype=torch.long)
        pred_cntx_idx = torch.tensor(self.pred_cntx_idx, dtype=torch.long)
        labels = torch.tensor(self.labels, dtype=torch.float32)
        label_idxes = torch.tensor(self.label_idxes, dtype=torch.long)
        pred_idx_in_gr = torch.tensor(self.pred_idx_in_gr, dtype=torch.long)
        entPairId = torch.tensor(self.entPairId, dtype=torch.long)
        pred_cntx_idx_end = torch.tensor(self.pred_cntx_idx_end, dtype=torch.long)
        linear_mask = torch.tensor([0,1] if self.pred_reversed else [1,0], dtype=torch.long)

        tensor_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "gr_idx": gr_idx,
            "pred_cntx_idx": pred_cntx_idx,
            "labels": labels,
            "label_idxes": label_idxes,
            "pred_idx_in_gr": pred_idx_in_gr,
            "entPairId": entPairId,
            "pred_cntx_idx_end": pred_cntx_idx_end,
            "linear_mask": linear_mask,
        }

        # Not used in the paper experiments.
        if self.hyp_ids:
            hyp_ids = torch.tensor(self.hyp_ids, dtype=torch.long)
            hyp_attention_masks = torch.tensor(self.hyp_attention_masks, dtype=torch.long)
            hyp_start_idxes = torch.tensor(self.hyp_start_idxes, dtype=torch.long)
            hyp_end_idxes = torch.tensor(self.hyp_start_idxes, dtype=torch.long)

            tensor_dict.update({
                "hyp_ids": hyp_ids,
                "hyp_attention_masks": hyp_attention_masks,
                "hyp_start_idxes": hyp_start_idxes,
                "hyp_end_idxes": hyp_end_idxes}
                               )

        return tensor_dict

class News_Iterable_Dataset(IterableDataset):

    def __init__(self, input_path, entgrAgg, data_mode, batch_size, preferred_num_labels, neg_ratio, max_seq_length, num_examples, mask_ents_prob, tokenizer, contextual_hyp, hard_negs, process_idx, num_processes, updated_len_loader, no_argord, triple_count_path):
        self.input_path = input_path
        self.entgrAgg = entgrAgg
        self.PMI = entgrAgg.PMI
        self.data_mode = data_mode
        self.batch_size = batch_size
        self.preferred_num_labels = preferred_num_labels
        self.neg_ratio = neg_ratio
        self.max_seq_length = max_seq_length
        self.num_examples = num_examples
        self.mask_ents_prob = mask_ents_prob
        self.cached_examples = [[] for _ in range(self.entgrAgg.num_graphs)]
        self.current_batch_size = batch_size # will only be smaller than batch_size when flushing the last batches.
        self.current_batch_types = ""
        self.tokenizer = tokenizer
        self.token2ids = dict()
        self.contextual_hyp = contextual_hyp
        self.hard_negs = hard_negs
        self.num_processes = num_processes
        self.process_idx = process_idx
        self.updated_len_loader = updated_len_loader
        self.no_argord = no_argord
        self.num_pos_labels = 0 # for stat purposes
        self.num_neg_labels = 0 # for stat purposes
        self.num_triple_mentions = 0 # for stat purposes
        if triple_count_path:
            self.triple_count_file = open(triple_count_path, 'w')
        else:
            self.triple_count_file = None
        self.finished = False

    def is_line_accepted(self, line_number):
        if self.data_mode == "all":
            return True
        elif self.data_mode == "train":
            return line_number % 20 != 0
        elif self.data_mode == "dev":
            return line_number %20 == 0 and line_number %40 == 0
        elif self.data_mode == "test":
            return line_number % 20 == 0 and line_number % 40 != 0
        else:
            raise Exception("not implemented data mode")

    def is_triple_accepted(self, triple):
        if "_1" in triple: # Let's make sure we have t_1#t_2, because we don't have the other way around in the triples.
            try:
                pred, arg1, arg2 = triple.split()
                pred_ss = pred.split("#")
                if pred_ss[1].endswith("_2"):
                    pred = "#".join([pred_ss[0], pred_ss[2], pred_ss[1]])
                    triple = " ".join([pred, arg2, arg1])
            except:
                print ("exception triple: " + str(triple))

        if not (triple in self.entgrAgg.train_triples or triple in self.entgrAgg.valid_triples or triple in self.entgrAgg.test_triples):
            print ("triple not in any set, will use it for train!" + str(triple))
        if self.data_mode == "all":
            return True
        elif self.data_mode == "dev":
            return triple in self.entgrAgg.valid_triples
        elif self.data_mode == "test":
            return triple in self.entgrAgg.test_triples
        else:
            return triple not in self.entgrAgg.valid_triples and triple not in self.entgrAgg.test_triples


    def tokenize(self, text, add_special_tokens=False):
        orig_tokens = text.split()
        bert_tokens = []
        orig_to_tok_map = []
        indexed_tokens = []
        if add_special_tokens:
            bert_tokens.append("[CLS]")
            indexed_tokens.extend(self.tokenizer.convert_tokens_to_ids(["[CLS]"]))
        for orig_token in orig_tokens:
            orig_to_tok_map.append(len(bert_tokens))
            this_tokens = self.tokenizer.tokenize(orig_token)
            bert_tokens.extend(this_tokens)
            indexed_tokens.extend(self.tokenizer.convert_tokens_to_ids(this_tokens))

        if add_special_tokens:
            indexed_tokens.extend(self.tokenizer.convert_tokens_to_ids(["[SEP]"]))
            bert_tokens.append("[SEP]")

        return bert_tokens, indexed_tokens, orig_to_tok_map


    #Find ent_tokens (e.g., barack obama) inside tokens
    # (e.g., [cls] barack oba ##ma visited hawaii . [sep]) => [[1,4]], i.e., tokens[1:4] = barack obama
    def find_sub_list(self, tokens, ent, ent_tokens):
        ret = []
        i = 0
        while i < len(tokens):
            if tokens[i] == ent_tokens[0]:
                cand = ""
                for j in range(i, len(tokens)):
                    cand += tokens[j].replace("##", "") + " "
                    if cand.strip() == ent:
                        ret.append([i,j+1])
                        i = j+1
                    if len(cand) >= len(ent):
                        break
            i += 1

        return ret

    def get_token_ids(self, type):# TODO (replace _ with space)
        if type not in self.token2ids:
            self.token2ids[type] = self.tokenize(type)[1]
        return self.token2ids[type]


    #It will mask input_ids in place. It returns the offset that should be applied to pred. Not used in the paper
    # experiments.
    def mask_ent(self, input_ids, ent, type, pred_cntx_idx):
        ent_tokens = ent.split(" ")
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        intervals = self.find_sub_list(tokens, ent, ent_tokens)
        type_ids = self.get_token_ids(type)

        #Now replace intervals with type_ids
        offset = 0
        pred_offset = 0
        for interval in intervals:
            input_ids[interval[0]+offset:interval[1]+offset] = type_ids

            print("interval: " + str(interval))
            this_offset = (len(type_ids) - (interval[1] - interval[0]))
            offset += this_offset
            if interval[0] < pred_cntx_idx:
                pred_offset += this_offset

            print("now offsets: " + str(offset) + " " + str(pred_offset))

        return pred_offset

    def __iter__(self):

        f = open(self.input_path)
        line_number = 0

        triple2count = defaultdict(int)

        for line in f:
            line_number += 1

            if not self.entgrAgg.triple_based_split: # split to train, valid, test based on lines
                if not self.is_line_accepted(line_number):
                    continue
            try:
                line_json = json.loads(line.strip())
            except:
                print("bad line: " + line)
                continue

            try:
                input_ids = line_json["bert_indexes"]
                tokens = None
                if _ADD_TOKENS:
                    tokens = line_json["bert_tokens"]

                # if line_number  >= 180: #
                #     break

                if line_number % 10 == 1:
                    print("Num all cached examples for " + self.data_mode + ": " + str(line_number) + " , " + str(
                        np.sum([len(x) for x in self.cached_examples])))
                    print("Stats, num triple mentions, num pos, neg labels: " + str(self.num_triple_mentions) + " " +
                          str(self.num_pos_labels) + " " + str(self.num_neg_labels))

                pred_in_context_infos = line_json["pred_in_context_infos"]

                for pred_in_context_info in pred_in_context_infos:

                    this_gr_idx = pred_in_context_info["gr_idx"]
                    if this_gr_idx % self.num_processes != self.process_idx:
                        continue
                    this_pred_idx_in_gr = pred_in_context_info["pred_idx_in_gr"]
                    this_pred_cntx_idx = pred_in_context_info["pred_idxes"][0]
                    this_pred_cntx_idx_end = pred_in_context_info["pred_idxes"][-1]
                    this_pos_labels = pred_in_context_info["pos_label_idxes"]

                    this_entPairId = 0
                    pred_reversed = pred_in_context_info["typed_pred"].split("#")[1].endswith("_2")
                    if self.no_argord:
                        pred_reversed = False

                    this_input_ids = input_ids

                    # The below block was not used in the paper experiments.
                    if self.entgrAgg.PMI or self.mask_ents_prob > 0:
                        try:
                            this_entPair = pred_in_context_info["entpair"]
                            pred = pred_in_context_info["typed_pred"]
                            types = "#".join(pred.split("#")[1:]).replace("_1", "").replace("_2", "")

                            ent1, ent2 = this_entPair.split("#")

                            if self.mask_ents_prob > 0:

                                pred_input_id0 = input_ids[this_pred_cntx_idx]
                                pred_first_tok = str(self.tokenizer.convert_ids_to_tokens(pred_input_id0))

                                should_assert = pred_first_tok not in ent1 and pred_first_tok not in ent2
                                type1, type2 = types.split("#")

                                print("pred, entpair: " + pred + " " + this_entPair + " " + str(
                                    pred_in_context_info["pred_idxes"]))
                                print("tokens before masking: " + str(self.tokenizer.convert_ids_to_tokens(input_ids)))

                                this_input_ids = deepcopy(input_ids)

                                # mask ent1:
                                if random.random() < self.mask_ents_prob:
                                    this_offset = self.mask_ent(this_input_ids, ent1, type1, this_pred_cntx_idx)
                                    this_pred_cntx_idx += this_offset
                                    this_pred_cntx_idx_end += this_offset

                                    print("tokens after masking1 of " + ent1 + " with " + type1 + " " + str(
                                        self.tokenizer.convert_ids_to_tokens(this_input_ids)))

                                print(str(
                                    self.tokenizer.convert_ids_to_tokens(
                                        this_input_ids[this_pred_cntx_idx])) + " vs1 " + str(
                                    self.tokenizer.convert_ids_to_tokens(pred_input_id0)))
                                if should_assert:
                                    assert this_input_ids[this_pred_cntx_idx] == pred_input_id0

                                # mask ent2
                                if random.random() < self.mask_ents_prob:
                                    this_offset = self.mask_ent(this_input_ids, ent2, type2, this_pred_cntx_idx)
                                    this_pred_cntx_idx += this_offset
                                    this_pred_cntx_idx_end += this_offset

                                    print("tokens after masking2 of " + ent2 + " with " + type2 + " " + str(
                                        self.tokenizer.convert_ids_to_tokens(this_input_ids)))

                                print(str(
                                    self.tokenizer.convert_ids_to_tokens(
                                        this_input_ids[this_pred_cntx_idx])) + " vs2 " + str(
                                    self.tokenizer.convert_ids_to_tokens(pred_input_id0)))
                                if should_assert:
                                    assert this_input_ids[this_pred_cntx_idx] == pred_input_id0

                                this_pred_cntx_idx = min(max(this_pred_cntx_idx, 0), len(this_input_ids) - 1)
                                this_pred_cntx_idx_end = min(max(this_pred_cntx_idx_end, 0), len(this_input_ids) - 1)

                                assert 0 <= this_pred_cntx_idx < len(this_input_ids)
                                assert 0 <= this_pred_cntx_idx_end < len(this_input_ids)



                        except:
                            print("exception in masking")
                            pass

                    this_entPair = pred_in_context_info["entpair"]
                    ent1, ent2 = this_entPair.split("#")
                    this_entPair = this_entPair.replace(" ", "_")
                    this_entPair = " ".join(this_entPair.split("#"))
                    this_pred = pred_in_context_info["typed_pred"]
                    this_triple = this_pred + " " + this_entPair

                    if self.entgrAgg.triple_based_split:
                        if not self.is_triple_accepted(this_triple):
                            continue

                    self.num_triple_mentions += 1
                    triple2count[this_triple] += 1

                    example = PredInCntxExample(
                        this_input_ids, this_gr_idx, this_pred_idx_in_gr, this_entPairId, ent1, ent2,
                        this_pred_cntx_idx, this_pred_cntx_idx_end, this_pos_labels, self.max_seq_length, pred_reversed,
                        tokens=tokens, triple=this_triple)
                    self.cached_examples[this_gr_idx].append(example)

                    if len(self.cached_examples[this_gr_idx]) >= self.batch_size:
                        self.entgrAgg.current_example_summaries = []
                        print("batch with types, offset, gr_idx: ", self.entgrAgg.entgraphs[this_gr_idx].types,
                              self.entgrAgg.entgraphs[this_gr_idx].gr_offset, this_gr_idx)
                        self.current_batch_size = self.batch_size
                        self.current_batch_types = self.entgrAgg.types_to_ordered_types[
                            self.entgrAgg.entgraphs[this_gr_idx].types]
                        self.prepare_batch(self.cached_examples[this_gr_idx], this_gr_idx)
                        this_examples = self.cached_examples[this_gr_idx]
                        self.cached_examples[this_gr_idx] = []
                        for example in this_examples:
                            if _ADD_TOKENS:
                                this_summary = " ".join([str(t) for t in example.tokens]) + " triple: " + example.triple
                                self.entgrAgg.current_example_summaries.append(this_summary)
                            yield example.get_tensor_dict()
            except Exception as e:
                print ("inside exception " + str(e))
                traceback.print_exc()


        f.close()

        print ("Flusing!")
        # FLUSH everything
        for this_gr_idx in range(self.entgrAgg.num_graphs):
            if len(self.cached_examples[this_gr_idx]) != 0:
                this_examples = self.cached_examples[this_gr_idx]
                print("batch with types, offset: ", self.entgrAgg.entgraphs[this_gr_idx].types,
                      self.entgrAgg.entgraphs[this_gr_idx].gr_offset)
                print ("flushing batch with size: " + str(len(this_examples)))
                self.current_batch_size = len(this_examples)
                self.current_batch_types = self.entgrAgg.types_to_ordered_types[
                    self.entgrAgg.entgraphs[this_gr_idx].types]
                self.cached_examples[this_gr_idx] = []
                this_examples_extended = []
                for i in range(self.batch_size):
                    this_examples_extended.append(this_examples[i % len(this_examples)])
                self.prepare_batch(this_examples_extended, this_gr_idx)
                for example in this_examples_extended:
                    yield example.get_tensor_dict()

        if self.triple_count_file:
            for triple, count in triple2count.items():
                self.triple_count_file.write(triple + ' ' + str(count)+ '\n')

        print ("setting self.finished to True")
        print("stats, num triple mentions, num pos, neg labels: " + str(self.num_triple_mentions) + " " +
              str(self.num_pos_labels) + " " + str(self.num_neg_labels))
        self.finished = True
        self.num_pos_labels = self.num_neg_labels = 0



    def __len__(self):
        # 8492840 for all
        # num_batches = 8500000 // self.batch_size
        # 35281225 // batch_size for NewsCrawl
        if not self.updated_len_loader:
            num_batches = self.num_examples // self.batch_size
            print("num examples in len: " + str(self.num_examples))
            print("num_batches in len: " + str(num_batches))
            return num_batches
        else:
            return self.num_examples

    # Get the longest phrase from the CCG relation.
    # relstr: string of format '(smile.1,smile.like.2)'
    # returns: 'smile like'. Not used in the paper experiments.
    def getPhraseFromCCGRel(self, relstr):
        negated = False

        # Check for negation in front: "NEG__(smile.1,smile.like.2)"
        if relstr.lower().startswith("neg__"):
            relstr = relstr[5:]
            negated = True
        # Check for other prefixes
        elif relstr.find("__(") != -1:
            st = relstr.find("__(")
            relstr = relstr[(st + 3):]

        # Remove parentheses. Split on comma.
        pair = relstr[1:-1].split(",")
        if len(pair) != 2:
            return ""

        # Removing trailing .1 or .2
        pred1 = pair[0][0:pair[0].rfind('.')]
        pred2 = pair[1][0:pair[1].rfind('.')]

        # Replace periods with spaces. Return the longest!
        phrase1 = pred1.replace('.', ' ')
        phrase2 = pred2.replace('.', ' ')
        if len(phrase2) >= len(phrase1):  # keep the longest
            phrase1 = phrase2
        if negated:  # check for negation
            phrase1 = "not " + phrase1
        return phrase1

    # Not used in the paper experiments.
    def tokenize_hyp(self, label_idx, entgraph):
        typed_pred = entgraph.preds[label_idx]
        try:
            pred, type1, type2 = typed_pred.replace("_1", "").replace("_2", "").split("#")
        except:
            pred = typed_pred.split("#")[0]
            type1, type2 = entgraph.types.split("#")

        ids = []
        pred = self.getPhraseFromCCGRel(pred)
        ids.extend(self.tokenizer.convert_tokens_to_ids(["[CLS]"]))
        ids.extend(self.tokenize(type1)[1])
        hyp_start_idx = len(ids)
        ids.extend(self.tokenize(pred)[1])
        hyp_end_idx = len(ids) - 1
        ids.extend(self.tokenize(type2)[1])
        ids.extend(self.tokenizer.convert_tokens_to_ids(["[SEP]"]))

        print ("ids: " + str(ids))

        print ("tokenized hyp: " + str(self.tokenizer.convert_ids_to_tokens(ids)))

        return ids, hyp_start_idx, hyp_end_idx

    # Not used in the paper experiments.
    def tokenize_hyps(self, all_label_idxes, entgraph):
        hyp_ids, hyp_attention_masks, hyp_start_idxes, hyp_end_idxes = [], [], [], []
        max_len = 0
        for label_idx in all_label_idxes:
            this_hyp_ids, this_hyp_start_idx, this_hyp_end_idx = self.tokenize_hyp(label_idx, entgraph)
            hyp_ids.append(this_hyp_ids)
            this_hyp_attention_masks = [1] * len(this_hyp_ids)
            max_len = max(max_len, len(this_hyp_ids))
            hyp_attention_masks.append(this_hyp_attention_masks)
            hyp_start_idxes.append(this_hyp_start_idx)
            hyp_end_idxes.append(this_hyp_end_idx)


        #Now, let's pad hyp_ids
        for i in range(len(hyp_ids)):
            padding_length = max_len - len(hyp_ids[i])
            pad_token = 0
            hyp_ids[i] += ([pad_token] * padding_length)
            hyp_attention_masks[i] += ([0] * padding_length)

        print ("hyp_ids: " + str(hyp_ids))
        print ("hyp_attention_masks: " + str(hyp_attention_masks))

        return hyp_ids, hyp_attention_masks, hyp_start_idxes, hyp_end_idxes


    # Set labels and pad input_ids
    # This will set a customized label array for the batch, based on all the pos_labels as well as random neg labels.
    def prepare_batch(self, examples, gr_idx):

        entgraph = self.entgrAgg.entgraphs[gr_idx]

        gr_size = entgraph.gr_size
        all_pos_labels = set()
        hard_neg_labels = set()
        for example in examples:
            self.num_pos_labels += len(example.pos_labels)
            all_pos_labels = all_pos_labels.union(set(example.pos_labels))
            if self.hard_negs and self.data_mode == "train":
                ent1_rels = entgraph.ent1ToRelIds[example.ent1]
                ent2_rels = entgraph.ent2ToRelIds[example.ent2]
                hard_neg_labels.update(ent1_rels)
                hard_neg_labels.update(ent2_rels)


        neg_label_cands = set(range(gr_size)).difference(all_pos_labels)

        if self.hard_negs and self.data_mode == "train":
            hard_neg_labels = hard_neg_labels.difference(all_pos_labels)
            neg_label_cands = neg_label_cands.difference(hard_neg_labels) # So, neg_label_cands are anything other than all_pos_labels and hard_neg_labels

        if self.data_mode == "all": # This means that we're building entgraphs, because there's no training, test split.
            num_labels = min(max(len(all_pos_labels), self.preferred_num_labels), gr_size)
        elif self.data_mode == "train":
            num_labels = min(max(int((1 + self.neg_ratio) * len(all_pos_labels)), self.preferred_num_labels), gr_size)
        else:
            num_labels = min(max(2 * len(all_pos_labels), self.preferred_num_labels), gr_size)

        num_neg_labels = num_labels - len(all_pos_labels)
        assert num_neg_labels >= 0

        if not (self.hard_negs and self.data_mode == "train"):
            all_neg_labels = set(random.sample(list(neg_label_cands), num_neg_labels))
        else:

            if len(hard_neg_labels) < num_neg_labels // 2:
                len1 = len(hard_neg_labels)
                len2 = num_neg_labels - len1
            elif len(neg_label_cands) < num_neg_labels - num_neg_labels // 2:
                len2 = len(neg_label_cands)
                len1 = num_neg_labels - len2
            else:
                len1 = num_neg_labels // 2
                len2 = num_neg_labels - len1

            all_neg_labels = set(random.sample(list(hard_neg_labels), len1)).union(random.sample(list(neg_label_cands), len2))

        self.num_neg_labels += len(all_neg_labels) * len(examples)
        all_label_idxes = sorted(list(all_pos_labels.union(all_neg_labels)))

        if self.contextual_hyp and self.data_mode == "train" and len(all_label_idxes) > 4000: # Memory consideration
            all_label_idxes = random.sample(all_label_idxes, 4000)

        max_len = max([len(example.input_ids) for example in examples])

        print ("max_len: ", max_len)

        for example in examples:
            example.set_labels(all_label_idxes, entgraph.gr_offset)
            example.pad(max_len)

        if self.contextual_hyp:
            hyp_ids, hyp_attention_masks, hyp_start_idxes, hyp_end_idxes = self.tokenize_hyps(all_label_idxes, entgraph)
            for example in examples:
                example.set_hyp_tokeniztion_info(hyp_ids, hyp_attention_masks, hyp_start_idxes, hyp_end_idxes)


class EntGraphAggregator:
    def __init__(self, trels_folder, PMIWeight=False, hard_negs=False, max_neighs = 1000, all_triples_path=None, num_fill=100, alpha=.5, write_output=False):
        self.trels_folder = trels_folder
        self.max_neighs = max_neighs
        self.PMI = PMIWeight
        self.hard_negs = hard_negs
        self.num_fill = num_fill
        self.alpha = alpha
        self.write_output = write_output
        self.predToEmb = dict()
        self.predToIdx = dict()
        self.types_to_ordered_types = dict()
        self.entgraphs = self.initiate_graphs()
        self.num_all_labels = np.sum(entgraph.gr_size for entgraph in self.entgraphs)
        self.triple_based_split = all_triples_path!=None
        if all_triples_path:
            self.train_triples, self.valid_triples, self.test_triples = self.read_all_triples(all_triples_path)
        print("num_all_labels: ", self.num_all_labels)
        self.num_graphs = len(self.entgraphs)
        self.current_example_summaries = [] # Just for logging purposes

    # Not used in the paper experiments.
    def get_init_emb_weights(self, embs_init_path):
        f = open(embs_init_path)
        embs = None
        for line in f:
            line = line.strip()
            ss = line.replace("[","").replace("]","").split()
            pred = ss[0]
            if embs is None:
                embs = np.ndarray((self.num_all_labels, len(ss) - 1))
            if pred not in self.predToIdx:
                print ("unknown pred!: " + pred)
            embs[self.predToIdx[pred], :] = [np.float(x) for x in ss[1:]]
            print ("pred, idx: ", pred, str(self.predToIdx[pred]))

        return embs

    def initiate_graphs(self):
        files = os.listdir(self.trels_folder)
        files = sorted(files)
        files = [f for f in files if f.endswith("_trels.txt")]
        entgraphs = [None for _ in range(len(files))]
        num_f = 0
        offset = 0
        for f in files:
            trels_file = open(self.trels_folder + "/" + f)
            ss_first = trels_file.readline().strip().split(",")

            this_gr_idx = np.int(ss_first[2].split(" ")[-1])
            this_types = f[:-10]
            this_preds = trels_file.readline().strip().split(" ")

            self.predToIdx.update({pred: i + offset for i, pred in enumerate(this_preds)})

            num_f += 1

            entgraph = EntGraph(this_types, this_preds, offset, self.max_neighs)

            types_ss = this_types.split("#")
            types_reverse = "#".join([types_ss[1], types_ss[0]])
            self.types_to_ordered_types[this_types] = this_types
            self.types_to_ordered_types[types_reverse] = this_types

            entgraphs[this_gr_idx] = entgraph
            offset += entgraph.gr_size

            currentTimeEntPair = ""

            if self.PMI or self.hard_negs:
                readTimeEntPair = False
                for line in trels_file:
                    line = line.strip()
                    if line == "":
                        readTimeEntPair = True
                    elif readTimeEntPair:
                        splitIdx = line.rfind(":")
                        currentTimeEntPair = line[:splitIdx]
                        count = np.int(line[splitIdx+1:].strip())
                        entgraph.add_timeEntPair(currentTimeEntPair, count)
                        readTimeEntPair = False
                    else:
                        splitIdx = line.rfind(":")
                        currentPred = line[:splitIdx]
                        count = np.int(line[splitIdx+1:].strip())
                        entgraph.update_predCount(currentPred, currentTimeEntPair, count)

            trels_file.close()

        return entgraphs


    def get_gr_offsets(self):
        gr_offsets = []
        offset = 0
        for gr_size in self.gr_sizes:
            gr_offsets.append(offset)
            offset += gr_size
        return gr_offsets

    def read_all_triples(self, all_triples_path):
        train_triples = self.read_triples(all_triples_path + "/train.txt")
        valid_triples = self.read_triples(all_triples_path + "/valid.txt")
        test_triples = self.read_triples(all_triples_path + "/test.txt")
        return train_triples, valid_triples, test_triples

    def read_triples(self, triples_path):
        ret = set()
        f = open(triples_path)
        for line in f:
            ss = line.strip().split("\t")
            triple = " ".join([ss[1], ss[0], ss[2]])
            ret.add(triple)

        return ret

    def write_pred_embs(self, embs_init_path):
        f_out_emb = open(embs_init_path, 'w')
        for pred in self.predToEmb:
            f_out_emb.write(pred + " " + str(np.array(self.predToEmb[pred].emb)) + "\n")
        f_out_emb.close()


    def update_entgraph(self, batch, ent_scores):

        ent_scores = ent_scores.numpy() # (bsz * num_labels)
        gr_idx = batch[2].numpy()[0] # (1)
        entgraph = self.entgraphs[gr_idx]
        labels = batch[4].numpy()
        label_idxes = batch[5].numpy()[0] # (num_labels)
        offset = entgraph.gr_offset
        label_idxes = [label_idx - offset for label_idx in label_idxes]
        pred_idx_in_grs = batch[6].numpy() # (batch_size)
        entPair_ids = batch[7].numpy()  # (batch_size)

        entgraph.update_scores(pred_idx_in_grs, label_idxes, entPair_ids, ent_scores, labels, self.current_example_summaries, self.num_fill, self.alpha, self.write_output)


    def set_self_score_one(self, ent_scores, batch):
        label_idxes = batch[5].numpy()[0]  # (num_labels)
        pred_idx_in_grs = batch[6].numpy()  # (batch_size)

        gr_idx = batch[2].numpy()[0]  # (1)
        entgraph = self.entgraphs[gr_idx]
        offset = entgraph.gr_offset

        pred_idx_to_idx = {label_idxes[i]:i for i in range(len(label_idxes))}


        for i, p in enumerate(pred_idx_in_grs):
            if i >= ent_scores.shape[0]:
                break
            ent_scores[i][pred_idx_to_idx[p + offset]] = 1

    def set_self_score_zero(self, ent_scores, labels, batch):
        label_idxes = batch[5].numpy()[0]  # (num_labels)
        pred_idx_in_grs = batch[6].numpy()  # (batch_size)

        gr_idx = batch[2].numpy()[0]  # (1)
        entgraph = self.entgraphs[gr_idx]
        offset = entgraph.gr_offset

        pred_idx_to_idx = {label_idxes[i]:i for i in range(len(label_idxes))}


        for i, p in enumerate(pred_idx_in_grs):
            if i >= ent_scores.shape[0]:
                break
            ent_scores[i][pred_idx_to_idx[p + offset]] = 0
            labels[i][pred_idx_to_idx[p + offset]] = 0

    def report_eval_scores(self, batch, ent_scores, current_batch_size):
        gr_idx = batch[2].numpy()[0]  # (1)
        entgraph = self.entgraphs[gr_idx]
        labels = batch[4].numpy()
        label_idxes = batch[5].numpy()[0]  # (num_labels)
        offset = entgraph.gr_offset
        label_idxes = [label_idx - offset for label_idx in label_idxes]
        pred_idx_in_grs = batch[6].numpy()  # (batch_size)
        entPair_ids = batch[7].numpy()  # (batch_size)

        entgraph.report_eval_scores(pred_idx_in_grs, label_idxes, entPair_ids, ent_scores, labels,
                               self.current_example_summaries, current_batch_size)


    def write_eval_entailment_info(self, batch, ent_scores, prem_emb, line_id, f_out):
        ent_scores = ent_scores.numpy()[0,:]  # (1 * num_labels)
        gr_idx = batch[2].numpy()[0]  # (1)
        entgraph = None
        if gr_idx!=-1:
            entgraph = self.entgraphs[gr_idx]
        label_idxes = batch[5].numpy()[0]  # (num_labels)
        if entgraph:
            offset = entgraph.gr_offset
            label_idxes = [label_idx - offset for label_idx in label_idxes]
        pred_idx_in_grs = batch[6].numpy()  # (batch_size)

        summary = self.current_example_summaries[0]
        split_idx = summary.rfind("triple:")
        prem = summary[:split_idx].strip().replace(" ##", "").replace("[CLS]", "").replace("[SEP]", "").strip()
        triple = summary[split_idx + len("triple: "):]

        prem_emb = prem_emb[0,:].tolist()
        print ("prem_emb: ", prem_emb)

        if entgraph:
            print ("ent_scores: " + str(ent_scores))
            print ("type ent_scores: " + str(type(ent_scores)))
            print ("label_idxes: " + str(label_idxes))
            eval_json = {"articleId": str(line_id//2), "lineId": str(line_id), "prem": prem,
                         "types": entgraph.types, "triple": triple, "emb": str(prem_emb),
                         "entailment_scores": entgraph.get_eval_entailment_scores(ent_scores, label_idxes)}
        else:
            types = "#".join(triple.split(" ")[0].split("#")[1:]).replace("_1","").replace("_2","")
            eval_json = {"articleId": str(line_id // 2), "lineId": str(line_id), "prem": prem,
                         "types": types, "triple": triple, "emb": str(prem_emb),
                         "entailment_scores": ""}
        json.dump(eval_json, f_out)
        f_out.write("\n")


    def write_graphs(self, entgraph_folder):
        if not os.path.isdir(entgraph_folder):
            os.mkdir(entgraph_folder)
        for graph in self.entgraphs:
            if sum([len(neighs) for neighs in graph.all_neighs]):
                graph.write_entgraph(entgraph_folder)


class NeighPred:
    def __init__(self, idx):
        self.idx = idx
        self.score = 0
        # self.count = 0

    def update_score(self, score):
        self.score += score
        # self.count += 1
        # self.score /= self.count

class EntGraph:
    def __init__(self, types, preds, offset, max_neighs):
        self.types = types
        self.preds = preds
        self.gr_offset = offset
        self.gr_size = len(preds)
        self.num_seen = [0 for _ in range(len(self.preds))]
        self.max_neighs = max_neighs
        self.all_neighs = [dict() for _ in range(len(self.preds))]
        self.predToId = {pred:i for i, pred in enumerate(self.preds)}
        self.timeEntpairToId = dict()
        self.timeEntPairCounts = list() # number of tuples with each timeEntPair
        self.entPairToId = dict()
        self.entPairs = list()
        self.entPairCounts = list()
        self.ent1ToRelIds = defaultdict(set) # entity in first slot to rel_ids
        self.ent2ToRelIds = defaultdict(set) # entity in second slot to rel_ids
        self.predEntPairIdToCount = defaultdict(int)
        self.predCounts = [0] * len(self.preds) # number of tuples with each predicate
        self.seen_pred_ids = set()
        self.numAllTuples = 0
        self.topEntScores = True

    def add_timeEntPair(self, timeEntPair, count):
        self.timeEntpairToId[timeEntPair] = len(self.timeEntpairToId)
        self.timeEntPairCounts.append(count)

        entPair = "#".join(timeEntPair.split("#")[1:])

        if entPair not in self.entPairToId:
            self.entPairToId[entPair] = len(self.entPairToId)
            self.entPairs.append(entPair)
            self.entPairCounts.append(0)
        self.entPairCounts[self.entPairToId[entPair]] += count



    def update_predCount(self, pred, timeEntPair, count):
        pred_id = self.predToId[pred]
        self.predCounts[pred_id] += count
        entPair = "#".join(timeEntPair.split("#")[1:])
        self.predEntPairIdToCount[str(self.predToId[pred])+"#"+str(self.entPairToId[entPair])] += count

        ent1, ent2 = entPair.split("#")
        self.ent1ToRelIds[ent1].add(pred_id)
        self.ent2ToRelIds[ent2].add(pred_id)

    def get_eval_entailment_scores(self, ent_scores, label_idxes):
        sortedIdxes = np.argsort(-ent_scores)
        ret = ""
        for rank, n in enumerate(sortedIdxes):
            ret += " " + self.preds[label_idxes[n]] + " " + str(ent_scores[n])
        return ret

    def update_scores(self, pred_idx_in_grs, label_idxes, entPair_ids, ent_scores, labels, summaries, num_fill, alpha, write_output):

        for p, idx in enumerate(pred_idx_in_grs):

            self_score = 1

            self.num_seen[idx] += self_score
            neighs = self.all_neighs[idx]
            self.seen_pred_ids.add(idx)

            if _ADD_TOKENS and write_output:
                print("\n" + summaries[p])

            if write_output:
                print("pred: " + self.preds[idx] + " self_score: " + str(self_score))
                print("neigh sortIdx: ")
            scores = ent_scores[p]
            sortedIdxes = np.argsort(-scores)


            #ORIGINAL SCORE BELOW

            if not self.topEntScores:
                sumScores = 0
                for n in range(len(label_idxes)):
                    if labels[p,n] == 1:
                        sumScores += ent_scores[p, n]


                for n, neigh_idx in enumerate(label_idxes):
                    if labels[p,n] == 0:
                        continue
                    if neigh_idx not in neighs:
                        neighs[neigh_idx] = NeighPred(neigh_idx)

                    neighs[neigh_idx].update_score(self_score * ent_scores[p, n] / sumScores) # Added on 23 April. MC score

            else:

                # TopEntScore code with the actual neighbors as well at most num_neighs new ones
                num_neighs = 0
                for rank, n in enumerate(sortedIdxes):
                    if labels[p, n] == 1:
                        num_neighs += 1

                selected_neighs = list()
                num_new_neighs = 0
                sumScores = 0
                for n in sortedIdxes:  # Look at the highest scores!
                    if labels[p, n] == 1 or len(selected_neighs) < num_fill:
                        selected_neighs.append(n)
                        if labels[p, n] == 0:
                            ent_scores[p, n] *= alpha
                        sumScores += ent_scores[p, n]
                    if labels[p, n] == 0:
                        num_new_neighs += 1

                for n in selected_neighs:
                    neigh_idx = label_idxes[n]
                    if write_output:
                        print("adding top scorer: " + self.preds[label_idxes[n]] + " " + str(
                            ent_scores[p, n]) + " label: " + str(labels[p, n]))

                    if neigh_idx not in neighs:
                        neighs[neigh_idx] = NeighPred(neigh_idx)

                    neighs[neigh_idx].update_score(self_score * ent_scores[p, n] / sumScores)  # Added on 23 April. MC score


    def report_eval_scores(self, pred_idx_in_grs, label_idxes, entPair_ids, ent_scores, labels, summaries, current_batch_size):

        for p, idx in enumerate(pred_idx_in_grs):

            if p >= current_batch_size:
                break

            if _ADD_TOKENS:
                print("\n" + summaries[p])

            print("pred: " + self.preds[idx])

            print("neigh sortIdx: ")
            scores = ent_scores[p]
            sortedIdxes = np.argsort(-scores)

            num_neighs = 0
            for rank, n in enumerate(sortedIdxes):
                if labels[p, n] == 1:
                    num_neighs += 1

            selected_neighs = list()
            num_new_neighs = 0
            for n in sortedIdxes:  # Look at the highest scores!
                if labels[p, n] == 1 or len(selected_neighs) < 100:
                    selected_neighs.append(n)
                if labels[p, n] == 0:
                    num_new_neighs += 1

            for n in selected_neighs:
                print("top scorer: " + self.preds[label_idxes[n]] + " " + str(
                    ent_scores[p, n]) + " label: " + str(labels[p, n]))



    def filter_neighs(self, neighs):
        if len(neighs) > self.max_neighs:
            neighs = sorted(neighs.values(), key=lambda n: n.score, reverse=True)[:self.max_neighs]
            neighs = {n.idx:n for n in neighs}
        return neighs


    def write_entgraph(self, entgraph_dir):
        N = len(self.seen_pred_ids)
        if N == 0:
            return
        fname = entgraph_dir + "/" + self.types + "_sim.txt"
        op = open(fname, 'w')
        op.write(self.types + " " + " num preds: " + str(N) + "\n")

        for idx, pred in enumerate(self.preds):
            neighs = self.all_neighs[idx]
            if len(neighs) == 0:
                continue

            for neigh in neighs.values():
                neigh.score /= self.num_seen[idx]

            neighs = sorted(neighs.values(), key=lambda n: n.score, reverse=True)


            op.write("predicate: " + pred + "\n")
            op.write("max num neighbors: " + str(len(neighs)) + "\n")
            op.write("\n")

            sum = 0
            op.write("contextualized sims\n")
            for neigh in neighs:
                op.write(self.preds[neigh.idx] + " " + str(neigh.score) + "\n")
                sum += neigh.score
            if sum <.999 or sum > 1.001: #
                print ("bug: " + pred + " " + str(sum))
            op.write("\n")
        op.close()

        print("results written for: ", fname)

# Used for loading an already learned entailment graph
class PreBuiltEntGraphCollection:
    def __init__(self, prebuilt_entgraph_dir, prebuilt_featIdx, prebuilt_simSuffix, entgrAgg):
        self.entgrAgg = entgrAgg
        self.types_to_graph = dict()
        args = argparse.ArgumentParser(description='graph args')
        args.CCG = True
        Graph.featIdx = args.featIdx = prebuilt_featIdx
        args.saveMemory = False
        args.threshold = -1.0

        prebuilt_entgraph_dir += "/"

        files = os.listdir(prebuilt_entgraph_dir)
        files = list(np.sort(files))

        num_f = 0

        for f in files:
            if num_f % 50 == 0:
                print("num prebuild_entgraph files: ", num_f)
            thisGpath = prebuilt_entgraph_dir + f

            if num_f == 10:
                break

            if prebuilt_simSuffix not in f or os.stat(thisGpath).st_size == 0:
                continue

            num_f += 1

            gr = Graph(gpath=thisGpath, args=args, lower=False)
            gr.set_Ws()

            types_ss = "#".join(gr.types).replace("_1", "").replace("_2", "").split("#")
            types1 = types_ss[0] + "#" + types_ss[1]
            print ("types1: " + str(types1))
            self.types_to_graph[types1] = gr

            types2 = types_ss[1] + "#" + types_ss[0]
            print("types2: " + str(types2))
            self.types_to_graph[types2] = gr


    def get_entailment_score(self, gr, i, j):
        try:
            if i != -1 and j != -1:
                return gr.get_w(i, j)
            return 0
        except:
            traceback.print_exc()
            print ("couldn't find entailment score for " + str(i) + " " + str(j))
            return 0


    def get_ent_scores(self, batch):
        gr_idx = batch[2].numpy()[0]  # (1)
        entgraph = self.entgrAgg.entgraphs[gr_idx]
        label_idxes = batch[5].numpy()[0]  # (num_labels)
        offset = entgraph.gr_offset
        label_idxes = [label_idx - offset for label_idx in label_idxes]
        pred_idx_in_grs = batch[6].numpy()  # (batch_size)
        bsz = pred_idx_in_grs.shape[0]
        num_labels = len(label_idxes)
        # labels = batch[4].numpy()

        entscores = np.zeros((bsz, num_labels))

        types = entgraph.types

        if types not in self.types_to_graph:
            print("doesn't have types: ", types)
        else:
            print ("has types: " + types)

        if types in self.types_to_graph:
            gr = self.types_to_graph[types]

            label_idxes_in_gr = [gr.pred2Node[entgraph.preds[label_idx]].idx if entgraph.preds[label_idx] in gr.pred2Node else -1 for label_idx in label_idxes]

            for p, pred_idx in enumerate(pred_idx_in_grs):
                pred1 = entgraph.preds[pred_idx]
                i = gr.pred2Node[pred1].idx if pred1 in gr.pred2Node else -1

                for l, label_idx in enumerate(label_idxes):
                    entscores[p, l] = self.get_entailment_score(gr, i, label_idxes_in_gr[l])

        return entscores
