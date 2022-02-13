from transformers import (
    BertTokenizer,
    BertConfig,
    BertModel,
    RobertaConfig,
    RobertaTokenizer,
    RobertaModel,
)


MODEL_CLASSES = {
    "bert": (BertConfig, BertModel, BertTokenizer),
    "roberta": (RobertaConfig, RobertaModel, RobertaTokenizer),
}

import torch
_DEBUG = False

class BertWrapper:
    def __init__(self, model_type, model_name):
        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]

        self.model = model_class.from_pretrained(
            model_name
        )
        self.tokenizer = tokenizer_class.from_pretrained(
            model_name
        )
        self.model.eval()

    def tokenize(self, text):
        orig_tokens = text.split()
        bert_tokens = []
        orig_to_tok_map = []
        bert_tokens.append("[CLS]")
        indexed_tokens = []
        indexed_tokens.extend(self.tokenizer.convert_tokens_to_ids(["[CLS]"]))
        for orig_token in orig_tokens:
            orig_to_tok_map.append(len(bert_tokens))
            this_tokens = self.tokenizer.tokenize(orig_token)
            bert_tokens.extend(this_tokens)
            indexed_tokens.extend(self.tokenizer.convert_tokens_to_ids(this_tokens))
        indexed_tokens.extend(self.tokenizer.convert_tokens_to_ids(["[SEP]"]))
        indexed_tokens = indexed_tokens
        bert_tokens.append("[SEP]")

        return bert_tokens, indexed_tokens, orig_to_tok_map

    def get_encoded_layer(self, tokens_tensor):

        with torch.no_grad():
            encoded_layer, _ = self.model(tokens_tensor) # , output_all_encoded_layers=False

        return encoded_layer[0]

    def get_tokens_emb(self, encoded_layer, start_idx, end_idx, only_start=True):

        emb = encoded_layer[0, start_idx, :]
        if not only_start:
            emb = (emb + encoded_layer[0, end_idx, :]) / 2

        return emb

    def get_tokens_emb_from_text(self, text, start_idx):
        bert_tokens, indexed_tokens, orig_to_tok_map = self.tokenize(text)
        cls_id = self.tokenizer.convert_tokens_to_ids(["[CLS]"])
        sep_id = self.tokenizer.convert_tokens_to_ids(["[SEP]"])
        this_indexed_tokens = cls_id + indexed_tokens + sep_id
        tokens_tensor = torch.tensor([this_indexed_tokens])
        this_encoder_layer = self.get_encoded_layer(tokens_tensor)

        start_idx = orig_to_tok_map[start_idx]

        return this_encoder_layer[start_idx, :]


def simpleNormalize(s):
    s = s.replace("_", " ")
    s = s.replace("-", " ")
    s = s.lower().strip()
    return s
