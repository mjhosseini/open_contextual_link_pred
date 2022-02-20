# coding=utf-8
# This code is adapted from Huggingface transformer's codebase
# (https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/modeling_bert.py) for the
# contextual link prediction task.
"""PyTorch BERT model for contextual link prediction."""

import torch
from torch import nn

from transformers import BertConfig, BertModel, BertPreTrainedModel

class BertForEntGraphs(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.hidden_size = config.hidden_size

        # The below line is a placeholder at the beginning, but will be filled in the rest of from_pretrained caller.
        self.bert = BertModel(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pred_embs = nn.Embedding(config.num_all_labels, config.hidden_size)

        self.linear_layer = nn.Linear(2 * config.hidden_size, config.hidden_size)

        self.init_weights()

        self.sigmoid = nn.Sigmoid()
        self.loss = torch.nn.BCELoss(reduction='none')

    def init_emb_weights(self, weights):
        self.pred_embs.weight.data.copy_(torch.from_numpy(weights))

    def forward(
        self,
        input_ids,
        attention_masks,
        pred_cntx_idxes,
        labels,
        label_idxes,
        pred_cntx_idxes_end = None,
        linear_masks = None,
        return_encoding = False
    ):
        r"""
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, max_seq_length)`):
            The input_ids corresponding to the tokens (around the predicates in context).
        attention_masks (:obj:`torch.LongTensor` of shape :obj:`(batch_size, max_seq_length)`):
            The attention masks.
        pred_cntx_idxes (:obj:`torch.LongTensor` of shape :obj:`(batch_size)`):
            The start index of the premise predicates in the context.
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, num_hyp)`):
            Whether each hyp (hypothesis) predicate is entailed (1) or not (0), given the prem (premise) predicate in
            context.
        label_idxes (:obj:`torch.LongTensor` of shape :obj:`(num_hyp)`):
            Index of each hyp predicate among all existing predicates of any type. It's the union of all hyp idxes for
            all predicates in the current batch.
        pred_cntx_idxes_end (:obj:`torch.LongTensor` of shape :obj:`(batch_size)`):
            The end index of the premise predicates in the context.
        linear_masks (:obj:`torch.LongTensor` of shape :obj:`(batch_size, 2)`):
            For each predicate in context, the linear mask is [0,1] if predicate types are reversed (e.g.,
            person_2#person_1), and [1,0] otherwise.
        return_encoding (boolean):
            Whether predicate encodings in context should be returned.



    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the inputs:
        outputs (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_labels)`):
            `num_labels` is the number of all predicates in all entailment graphs.
            The outputs correspond to the contextual link prediction scores.
        loss (:obj:`torch.FloatTensor` of shape `(1,)`, `optional`, returned when :obj:`labels` is provided):
            Classification loss.
        prem_emb (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, hidden_size)`, `optional`, returned when
            return_encoding=True):
            The encodings of predicates in the context.

        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_masks,
        )

        encoded_layer = outputs[0]

        encoded_layer = self.dropout(encoded_layer)

        batch_size = input_ids.shape[0]

        #bsz => bsz,1 => (bsz, 1, hdz)
        pred_cntx_idxes = pred_cntx_idxes.unsqueeze(1).repeat(1, self.config.hidden_size).view(batch_size, 1, self.config.hidden_size)

        # (bsz, seq_len, hdz) => (bsz, hdz)
        prem_emb = torch.gather(encoded_layer, 1, pred_cntx_idxes).squeeze(1)

        # below code is for (end + start)/2
        pred_cntx_idxes_end = pred_cntx_idxes_end.unsqueeze(1).repeat(1, self.config.hidden_size).view(batch_size, 1, self.config.hidden_size)
        prem_emb_end = torch.gather(encoded_layer, 1, pred_cntx_idxes_end).squeeze(1)

        prem_emb = (prem_emb + prem_emb_end) / 2

        # The below code is for type_1#type_2 vs type_2#type_1 projections, e.g., loc_1#loc_2 vs loc_2#loc_1.
        prem_emb = prem_emb.repeat(1,2)
        linear_masks = linear_masks.view(-1,1).repeat(1,self.hidden_size).view(batch_size, 2*self.hidden_size)

        prem_emb = prem_emb * linear_masks
        prem_emb = self.linear_layer(prem_emb)

        label_idxes = label_idxes[0] # Same thing across the whole batch

        #(num_labels, hdz)
        hyp_embs = self.pred_embs(label_idxes)

        #(bsz, hdz) * (hdz, num_labels) => (bsz, num_labels)
        logits = torch.mm(prem_emb, hyp_embs.transpose(1,0))

        scores = self.sigmoid(logits)

        outputs = (scores,)

        if labels is not None:
            loss = self.loss(scores.view(-1), labels.view(-1))
            loss = loss.mean()
            outputs = (loss,) + outputs

        if return_encoding:
            outputs = (prem_emb,) + outputs

        return outputs


