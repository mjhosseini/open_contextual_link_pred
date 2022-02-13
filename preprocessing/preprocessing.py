# -*- coding: utf-8 -*-

import os
import sys
import argparse
sys.path.append("..")
sys.path.append(".")
from collections import defaultdict
import json
import utils.util as util
import traceback
from datetime import datetime
import torch
import numpy as np
np.set_printoptions(linewidth=np.inf)
import concurrent.futures

class TypedRelsInfoCollection:

    def __init__(self, rels_folder, news_input_path, news_bert_input_path, window_size, only_start, model_type, model_name):
        self.rels_folder = rels_folder
        self.news_input_path = news_input_path
        self.news_bert_input_path = news_bert_input_path
        self.window_size = window_size
        self.only_start = only_start
        self.model_type = model_type
        self.model_name = model_name
        self.bert_wrapper = util.BertWrapper(model_type, model_name)
        self.date_format = "%b_%d,_%Y"
        self.min_date = None
        self.num_all_triples = 0
        self.num_all_triples_in_context = 0
        self.num_all_unique_triples = 0
        self.num_pos_labels = 0
        self.num_found_context = 0
        self.typesToTypedRelsInfo = dict()
        self.entToType = dict()
        self.all_triplesToCount = defaultdict(int)
        self.all_triplesToCount_in_context = defaultdict(int)
        self.timeIdToEntpairToPreds = defaultdict(lambda: defaultdict(set))
        self.entpairToTimeIds = defaultdict(list)
        self.typesToOrderedTypes = dict()
        self.predToEmb = dict()
        self.line_bert_inputs = list()
        self.writing_thread_idx = 0


    def build_bert_input_corpus(self):
        print("reading typed_rels")
        self.read_all_typed_rels()

        print ('num all triples: ', self.num_all_triples)
        print('num all unique triples: ', self.num_all_unique_triples)

        print("scanning the news corpus and populating timeEntpair to predicate counts")
        self.populate_timeEntpairToPredCounts()

        print('num all triples in context: ', self.num_all_triples_in_context)

        print("writing timed_rels")

        self.write_all_timed_rels()

        print("setting timeId to EntPairToPred")
        self.set_timeIdToEntPairToPred()

        print ("writing bert_input json and pre-trained embeddings")
        self.write_entgraph_bert_input()

        print ("num found context: " + str(self.num_found_context))
        print ("num pos labels " + str(self.num_pos_labels))


    def get_type(self, ent):
        if ent in self.entToType:
            return self.entToType[ent]
        else:
            return "thing"

    def read_all_typed_rels(self):
        print("rels_folder: " + self.rels_folder)
        files = os.listdir(self.rels_folder)
        files = sorted(files)
        num_f = 0
        for f in files:
            if not f.endswith("_rels.txt"):
                continue
            this_typedRelsInfo = TypedRelsInfo(self.rels_folder + f, num_f)
            num_f += 1
            self.entToType.update(this_typedRelsInfo.entToType)
            for triple in this_typedRelsInfo.triplesToCount:
                self.all_triplesToCount[triple] += this_typedRelsInfo.triplesToCount[triple]
            self.typesToTypedRelsInfo[this_typedRelsInfo.types] = this_typedRelsInfo
            types2 = this_typedRelsInfo.type2 + "#" + this_typedRelsInfo.type1
            self.typesToTypedRelsInfo[types2] = this_typedRelsInfo
            self.typesToOrderedTypes[this_typedRelsInfo.types] = this_typedRelsInfo.types
            self.typesToOrderedTypes[types2] = this_typedRelsInfo.types
            self.num_all_triples += this_typedRelsInfo.num_triples
            self.num_all_unique_triples += len(this_typedRelsInfo.triples)


    def read_news_input_json_lines(self):
        f = open(self.news_input_path)
        line_number = 0
        for line in f:
            try:
                line_json = json.loads(line.strip())
            except:
                print("bad line: " + line)
                continue

            if (line_number + 1) % 10000 == 0:
                print('line_number: ' + str(line_number))

            line_number += 1

            triples = line_json["rels"]
            timestamp = "_".join(line_json["date"].split(" ")[:3])
            tokens = line_json["tokens"]
            # This is not space. It's a special character. Inconsistency between bert split and the one done in Java.
            tokens = tokens.replace(" ","-")
            articleId = line_json["articleId"]
            lineId = line_json["lineId"]
            triple_infos = list()
            for triple in triples:
                triple = triple["r"]
                triple_ss = triple[1:-1].split("::")
                pred, ent1, ent2 = triple_ss[:3]
                pred_idxes = triple_ss[6]

                ent1_idxes = triple_ss[7]
                ent2_idxes = triple_ss[8]

                ent1 = util.simpleNormalize(ent1)
                ent2 = util.simpleNormalize(ent2)

                type1 = self.get_type(ent1)
                type2 = self.get_type(ent2)

                triple_infos.append((triple, pred, pred_idxes, ent1_idxes, ent2_idxes, ent1, ent2, type1, type2))

            yield articleId, lineId, tokens, timestamp, triple_infos

        f.close()

    def populate_timeEntpairToPredCounts(self):

        for line_info in self.read_news_input_json_lines():

            _, _, _, timestamp, triple_infos = line_info

            for triple_info in triple_infos:

                triple, pred, _, _, _, ent1, ent2, type1, type2 = triple_info

                types = type1 + "#" + type2

                if types not in self.typesToTypedRelsInfo:
                    continue

                this_typedRelsInfo = self.typesToTypedRelsInfo[types]

                # We'll need entPair in the order of its types for timeEntpair, so we'll have entailments with reverse order of args
                reversedOrder = self.typesToOrderedTypes[types] != types

                if reversedOrder:
                    timeEntpair = "#".join([timestamp, ent2, ent1])
                else:
                    timeEntpair = "#".join([timestamp, ent1, ent2])

                date = datetime.strptime(timestamp, self.date_format)

                if not self.min_date or date < self.min_date:
                    self.min_date = date

                if type1 != type2:
                    typed_pred = "#".join(([pred, type1, type2]))
                    typed_triple = "#".join([typed_pred, ent1, ent2])
                    if typed_triple not in this_typedRelsInfo.triples:  # This triple has been removed while applying the cutoffs.
                        continue
                    self.num_all_triples_in_context += 1
                    self.all_triplesToCount_in_context[typed_triple] += 1
                    this_typedRelsInfo.add_timestamped_triple(timeEntpair, typed_pred)
                else:
                    typed_pred1 = "#".join(([pred, type1 + "_1", type2 + "_2"]))
                    typed_triple1 = "#".join([typed_pred1, ent1, ent2])
                    if typed_triple1 in this_typedRelsInfo.triples:  # This triple has been removed while applying the cutoffs.
                        this_typedRelsInfo.add_timestamped_triple(timeEntpair, typed_pred1)
                        self.num_all_triples_in_context += 1
                        self.all_triplesToCount_in_context[typed_triple1] += 1

                    typed_pred2 = "#".join(([pred, type1 + "_2", type2 + "_1"]))
                    typed_triple2 = "#".join([typed_pred2, ent2, ent1])
                    if typed_triple2 in this_typedRelsInfo.triples:  # This triple has been removed while applying the cutoffs.
                        timeEntpair2 = "#".join([timestamp, ent2, ent1])
                        this_typedRelsInfo.add_timestamped_triple(timeEntpair2, typed_pred2)
                        self.num_all_triples_in_context += 1
                        self.all_triplesToCount_in_context[typed_triple2] += 1


    def write_all_timed_rels(self):
        for this_typedRelsInfo in self.typesToTypedRelsInfo.values():
            this_typedRelsInfo.write_timed_rels(self.rels_folder)

    def get_timeId(self, timestamp):
        date = datetime.strptime(timestamp, self.date_format)
        timeId = (date - self.min_date).days
        return timeId

    def set_timeIdToEntPairToPred(self):
        for this_typedRelsInfo in self.typesToTypedRelsInfo.values():
            for timeEntpair in this_typedRelsInfo.timeEntpairToPredCounts:
                ss = timeEntpair.split("#")
                entpair = ss[1] + "#" + ss[2]
                timestamp = ss[0]
                timeId = self.get_timeId(timestamp)
                self.timeIdToEntpairToPreds[timeId][entpair].update(this_typedRelsInfo.timeEntpairToPredCounts[timeEntpair].keys())
                self.entpairToTimeIds[entpair].append(timeId)



    def get_pred_in_context_info(self, typed_pred, types, orig_pred_idxes, orig_ent1_idxes, orig_ent2_idxes,
                                 orig_to_tok_map, timestamp, entpair):
        this_typedRelsInfo = self.typesToTypedRelsInfo[types]
        gr_idx = this_typedRelsInfo.gr_idx
        pred_idx_in_gr = this_typedRelsInfo.predToidx[typed_pred]
        orig_pred_idxes = [int(x) for x in orig_pred_idxes.split("_")]
        orig_ent1_idxes = [int(x) for x in orig_ent1_idxes.split("_")]
        orig_ent2_idxes = [int(x) for x in orig_ent2_idxes.split("_")]
        try:
            pred_idxes = [orig_to_tok_map[x] for x in orig_pred_idxes] #some rare error in parsing due to __
            ent1_idxes = [orig_to_tok_map[x] for x in orig_ent1_idxes]
            ent2_idxes = [orig_to_tok_map[x] for x in orig_ent2_idxes]
        except:
            traceback.print_exc()
            return None
        num_labels = len(this_typedRelsInfo.preds)

        center_timeId = self.get_timeId(timestamp)
        pos_label_idxes = []

        for timeId in range(center_timeId - self.window_size, center_timeId + self.window_size + 1):
            if timeId in self.timeIdToEntpairToPreds and entpair in self.timeIdToEntpairToPreds[timeId]:
                neigh_typed_preds = self.timeIdToEntpairToPreds[timeId][entpair]
                neigh_typed_preds = [pred for pred in neigh_typed_preds if pred in this_typedRelsInfo.predToidx]
                pos_label_idxes.extend(
                    [this_typedRelsInfo.predToidx[neigh_typed_pred] for neigh_typed_pred in neigh_typed_preds])

        assert pred_idx_in_gr in pos_label_idxes

        self.num_pos_labels += len(pos_label_idxes)

        if len(pos_label_idxes) > 1:
            self.num_found_context += 1

        #Let's have human-readable format in the output json file
        fixed_order_entPair = entpair
        reversedOrder = types != self.typesToOrderedTypes[types]
        if reversedOrder:
            ss = entpair.split("#")
            fixed_order_entPair = ss[1] + "#" + ss[0]

        pred_in_context_info = {"typed_pred" : typed_pred, "entpair" : fixed_order_entPair, "gr_idx" : gr_idx,
                                "pred_idx_in_gr" : pred_idx_in_gr, "pred_idxes" : pred_idxes, "ent1_idxes" : ent1_idxes,
                                "ent2_idxes" : ent2_idxes, "num_labels" : num_labels,
                                "pos_label_idxes" : pos_label_idxes}

        return pred_in_context_info

    def write_entgraph_bert_input(self):
        f_out = open(self.news_bert_input_path, 'w')

        num_threads = 1
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_threads) as executor:
            thread_idxes = [idx for idx in range(num_threads)]
            num_threads_list = [num_threads for _ in range(num_threads)]
            f_outs = [f_out for _ in range(num_threads)]
            num_updates = 0
            for _ in executor.map(self.write_entgraph_bert_input_thread, f_outs, thread_idxes,
                                           num_threads_list):
                print("comupte done")
                # num_updates += num_update

        self.flush_line_bert_inputs(f_out)
        f_out.close()

    def write_entgraph_bert_input_thread(self, f_out, thread_idx, num_threads):

        for line_info in self.read_news_input_json_lines():

            articleId, lineId, tokens, timestamp, triple_infos = line_info
            lineId = np.int(lineId)

            if lineId % num_threads != thread_idx:
                continue

            if lineId % 10000 == thread_idx:
                print ("lineId, thread_idx:", lineId, thread_idx)

            bert_tokens, indexed_tokens, orig_to_tok_map = self.bert_wrapper.tokenize(tokens)
            pred_in_context_infos = list()

            for triple_info in triple_infos:

                triple, pred, orig_pred_idxes, orig_ent1_idxes, orig_ent2_idxes, ent1, ent2, type1, type2 = triple_info

                types = type1 + "#" + type2

                if types not in self.typesToTypedRelsInfo:
                    continue

                this_typedRelsInfo = self.typesToTypedRelsInfo[types]

                if type1 != type2:
                    typed_pred = "#".join(([pred, type1, type2]))
                    typed_triple = "#".join([typed_pred, ent1, ent2])
                    if typed_triple not in this_typedRelsInfo.triples:  # This triple has been removed while applying the cutoffs.
                        continue

                    #We'll need entPair in the order of its types for timeEntpair, so we'll have entailments with reverse order of args
                    reversedOrder = types != self.typesToOrderedTypes[types]
                    if reversedOrder:
                        entPair = ent2 + "#" + ent1
                        #No need to swap orig_ent1_idxes and orig_ent2_idxes. Because at the end, we will swap ent2 and
                        # ent1 again for human-readability. This swap is just for our lookup.
                    else:
                        entPair = ent1 + "#" + ent2

                    pred_in_context_info = self.get_pred_in_context_info(
                        typed_pred, types, orig_pred_idxes, orig_ent1_idxes, orig_ent2_idxes,
                        orig_to_tok_map, timestamp, entPair)

                    if pred_in_context_info:
                        pred_in_context_infos.append(pred_in_context_info)
                    else:
                        print ("exception for ")
                        print ("tokens: ", tokens)
                        print ("bert tokens: ", bert_tokens)
                        print ("orig_to_tok_map:", orig_to_tok_map)

                else:

                    typed_pred1 = "#".join(([pred, type1 + "_1", type2 + "_2"]))
                    typed_triple1 = "#".join([typed_pred1, ent1, ent2])
                    if typed_triple1 in this_typedRelsInfo.triples:  # This triple has been removed while applying the cutoffs.
                        pred_in_context_info1 = self.get_pred_in_context_info(
                            typed_pred1, types, orig_pred_idxes, orig_ent1_idxes, orig_ent2_idxes, orig_to_tok_map,
                            timestamp, ent1 + "#" + ent2)

                        if pred_in_context_info1:
                            pred_in_context_infos.append(pred_in_context_info1)
                        else:
                            print("exception for ")
                            print("tokens: ", tokens)
                            print("bert tokens: ", bert_tokens)
                            print("orig_to_tok_map:", orig_to_tok_map)

                    typed_pred2 = "#".join(([pred, type1 + "_2", type2 + "_1"]))
                    typed_triple2 = "#".join([typed_pred2, ent2, ent1])

                    if typed_triple2 in this_typedRelsInfo.triples:  # This triple has been removed while applying the cutoffs.

                        pred_in_context_info2 = self.get_pred_in_context_info(
                            typed_pred2, types, orig_pred_idxes, orig_ent2_idxes, orig_ent1_idxes, orig_to_tok_map,
                            timestamp, ent2 + "#" + ent1)

                        if pred_in_context_info2:
                            pred_in_context_infos.append(pred_in_context_info2)
                        else:
                            print("exception for ")
                            print("tokens: ", tokens)
                            print("bert tokens: ", bert_tokens)
                            print("orig_to_tok_map:", orig_to_tok_map)

            if pred_in_context_infos:
                line_bert_input = {"articleId": articleId, "date": timestamp, "lineId": str(lineId), "bert_tokens": bert_tokens,
                                   "bert_indexes": indexed_tokens, "pred_in_context_infos": pred_in_context_infos}

                self.line_bert_inputs.append(line_bert_input)

                if thread_idx == self.writing_thread_idx and len(self.line_bert_inputs) >= 10:
                    self.flush_line_bert_inputs(f_out)
                    self.writing_thread_idx = (self.writing_thread_idx + 1) % num_threads



    def flush_line_bert_inputs(self, f_out):
        while len(self.line_bert_inputs) > 0:
            json.dump(self.line_bert_inputs.pop(0), f_out)
            f_out.write("\n")


    def compute_embeddings_line(self, line_bert_input):
        pred_in_context_infos = line_bert_input["pred_in_context_infos"]
        indexed_tokens = line_bert_input["bert_indexes"]

        cls_id = self.bert_wrapper.tokenizer.convert_tokens_to_ids(["[CLS]"])
        sep_id = self.bert_wrapper.tokenizer.convert_tokens_to_ids(["[SEP]"])
        max_len = 40

        indexed_tokens = indexed_tokens[1:-1] # trim cls and sep for now

        for i in range(len(indexed_tokens) // max_len + 1):
            this_indexed_tokens  = cls_id + indexed_tokens[i*max_len: min([(i+1)*max_len, len(indexed_tokens)])] + sep_id
            tokens_tensor = torch.tensor([this_indexed_tokens])
            this_encoder_layer = self.bert_wrapper.get_encoded_layer(tokens_tensor)

            for pred_in_context_info in pred_in_context_infos:
                pred_idxes = pred_in_context_info["pred_idxes"]
                if pred_idxes[-1] < (i+1)*max_len + 1 and pred_idxes[0] >= i*max_len + 1:
                    pred_idxes = [idx - i*max_len for idx in pred_idxes]
                    pred = pred_in_context_info["typed_pred"]
                    emb = self.bert_wrapper.get_tokens_emb(this_encoder_layer,pred_idxes[0], pred_idxes[-1], self.only_start)
                    if pred not in self.predToEmb:
                        self.predToEmb[pred] = PredEmb(pred, emb)
                    else:
                        self.predToEmb[pred].update_emb(emb)

class PredEmb:
    def __init__(self, pred, emb):
        self.pred = pred
        self.emb = emb
        self.count = 1

    def update_emb(self, emb):
        self.emb = (self.emb * self.count + emb) / (self.count + 1)
        self.count += 1


class TypedRelsInfo:
    def __init__(self, fname, gr_idx):
        self.fname = fname
        self.gr_idx = gr_idx
        self.types = self.type1 = self.type2 = None
        self.predToidx = dict()
        self.preds = list()
        self.triples = set()
        self.triplesToCount = defaultdict(int)
        self.num_triples = 0 #read from _rel files
        # self.num_triples_in_context = 0 #calculated based on triples matched in context
        self.entToType = dict()
        self.timeEntpairToPredCounts = defaultdict(lambda: defaultdict(int))
        self.read_typed_rels()

    def set_types(self, types):
        self.types = types
        self.type1, self.type2 = self.types.split("#")

    def add_predicate(self, pred):
        self.predToidx[pred] = len(self.preds)
        self.preds.append(pred)

    # we assume that pred, entpair is already in the set of acceptable triples
    def add_timestamped_triple(self, timeEntpair, pred):
        self.timeEntpairToPredCounts[timeEntpair][pred] += 1
        # self.num_triples_in_context += 1

    # returns typedRelsInfo, triples
    def read_typed_rels(self):
        f = open(self.fname)

        currentPred = ""
        reverseOrder = False

        for l in f:
            l = l.strip()
            if l.startswith("types:"):  # first line, let's continue!
                self.set_types(l.split(",")[0][7:])
            elif l.startswith("predicate:"):
                currentPred = l.split()[1]
                ss = currentPred.split("#")
                self.add_predicate(currentPred)
                try:
                    reverseOrder = (ss[1].replace("_1", "").replace("_2", "") != self.type1)
                except:
                    print ("bad relation: " + str(currentPred))
                    pass

            elif l == "":
                continue
            elif l.startswith("inv idx of"):
                break
            else:
                cidx = l.rfind(":")

                entPair = l[:cidx]
                args = entPair.split("#")

                count = int(float(l[cidx + 2:]))
                self.num_triples += count

                #based on the file format. Irrespective of reverseOrder
                self.entToType.update({args[0]: self.type1, args[1]: self.type2})

                if reverseOrder:
                    triple = "#".join([currentPred,args[1],args[0]])
                else:
                    triple = "#".join([currentPred, args[0], args[1]])

                self.triples.add(triple)
                self.triplesToCount[triple] += count

        f.close()

    def write_timed_rels(self, rels_folder):
        f = open(rels_folder + self.types + "_trels.txt", "w")
        f.write("types: " + self.types + ", num preds: " + str(len(self.preds)) + ", gr_idx: " + str(self.gr_idx) + "\n")
        preds_line = " ".join(self.preds)
        f.write(preds_line + "\n\n")

        for timeEntpair in self.timeEntpairToPredCounts:
            predCounts = self.timeEntpairToPredCounts[timeEntpair]
            sumCounts = sum(predCounts.values())
            f.write(timeEntpair + ": " + str(sumCounts) + "\n" + "\n".join([pred + ": " + str(predCounts[pred]) for pred in predCounts]) + "\n\n")

        f.close()


def create_entgraph_bert_input(rels_folder, news_json_path, news_bert_input_path, window_size, only_start, model_type, model_name):
    typedRIC= TypedRelsInfoCollection(rels_folder, news_json_path, news_bert_input_path, window_size, only_start, model_type, model_name)
    typedRIC.build_bert_input_corpus()

parser = argparse.ArgumentParser()
parser.add_argument(
        "--rels_folder",
        default=None,
        type=str,
        required=True,
        help="A directory containing predicate-argument structures (relations and entity pairs) for each type pair",
    )
parser.add_argument(
        "--input_path",
        default=None,
        type=str,
        required=True,
        help="A file containing the CCG-parsed corpus: predicate-argument structures and their indices in the text",
    )
parser.add_argument(
        "--output_path",
        default=None,
        type=str,
        required=True,
        help="The output data dir. It will contain a json file containing the parsed triple mentions and their"
             "contexts.",
    )
parser.add_argument(
        "--window_size",
        default=5000,
        type=int,
        required=True,
        help="For each entity-pair in context, the list of all relations that hold between the two entity pairs are "
             "stored in the output file. We only store those relations in an interval around the context's timestamp."
             "In all the experiments, the interval is set to infinity so that all the relations that hold between the"
             "entity-pair will be considered. Smaller values didn't improve the final contextual link prediction"
             "results.",
    )

parser.add_argument(
        "--model_type",
        default="bert",
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(util.MODEL_CLASSES.keys()),
    )

parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name",
    )

args = parser.parse_args()
# create_entgraph_bert_input("../../gfiles/typed_rels_aida_figer_3_3_f_copy/", "../../../java/entgraph/news_gen14.json", "../../../java/entgraph/news_roberta_large_input14_all.json", 100000, True, "roberta", "roberta-large")
create_entgraph_bert_input("../contextual_data/typed_rels/", "../contextual_data/news_gen.json", "../contextual_data/news_bert_input2.json", 5000, True, "bert", "bert-base-uncased")
create_entgraph_bert_input(args.rels_folder, args.input_path, args.output_path, args.window_size, True, args.model_type,
                           args.model_name_or_path)
