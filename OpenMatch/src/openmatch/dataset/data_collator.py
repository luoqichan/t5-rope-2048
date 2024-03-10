# Adapted from Tevatron (https://github.com/texttron/tevatron)

from dataclasses import dataclass

import torch
from transformers import DataCollatorWithPadding, DefaultDataCollator


def concatenate_tensors_gradcache_fusion(original_tensor, fusion):
        N, _ = original_tensor.shape
        new_tensor = original_tensor[:N//fusion*fusion].reshape(N//fusion, fusion, -1)
        concatenated_tensor = torch.cat(tuple(new_tensor.unbind(1)), dim=1)
        return concatenated_tensor

@dataclass
class QPCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    max_q_len: int = 32
    max_p_len: int = 128
    fusion: int = None

    def __call__(self, features):
        qq = [f["query_"] for f in features]
        dd = [f["passages"] for f in features]
        cc = [f["c_passages"] for f in features]

        if isinstance(qq[0], list):
            qq = sum(qq, [])
        if isinstance(dd[0], list):
            dd = sum(dd, [])
        if isinstance(cc[0], list):
            cc = sum(cc, [])

        q_collated = self.tokenizer.pad(
            qq,
            padding='max_length',
            max_length=self.max_q_len,
            return_tensors="pt",
        )
        d_collated = self.tokenizer.pad(
            dd,
            padding='max_length',
            max_length=self.max_p_len,
            return_tensors="pt",
        )

        c_collated = self.tokenizer.pad(
            cc, 
            padding='max_length',
            max_length=self.max_p_len,
            return_tensors="pt"
        )

        if self.fusion:
            d_collated['input_ids'] = concatenate_tensors_gradcache_fusion(d_collated['input_ids'], self.fusion)
            d_collated['attention_mask'] = concatenate_tensors_gradcache_fusion(d_collated['attention_mask'], self.fusion)

            c_collated['input_ids'] = concatenate_tensors_gradcache_fusion(c_collated['input_ids'], self.fusion)
            c_collated['attention_mask'] = concatenate_tensors_gradcache_fusion(c_collated['attention_mask'], self.fusion)

        return q_collated, d_collated, c_collated


@dataclass
class PairCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    max_q_len: int = 32
    max_p_len: int = 128

    def __call__(self, features):
        pos_pairs = [f["pos_pair"] for f in features]
        neg_pairs = [f["neg_pair"] for f in features]

        if isinstance(pos_pairs[0], list):
            pos_pairs = sum(pos_pairs, [])
        if isinstance(neg_pairs[0], list):
            neg_pairs = sum(neg_pairs, [])

        pos_pair_collated = self.tokenizer.pad(
            pos_pairs,
            padding='max_length',
            max_length=self.max_q_len + self.max_p_len + 2,
            return_tensors="pt",
        )
        neg_pair_collated = self.tokenizer.pad(
            neg_pairs,
            padding='max_length',
            max_length=self.max_q_len + self.max_p_len + 2,
            return_tensors="pt",
        )

        return pos_pair_collated, neg_pair_collated


@dataclass
class PairwiseDistillationCollator(DataCollatorWithPadding):

    max_q_len: int = 32
    max_p_len: int = 128

    def __call__(self, features):
        qq = [f["query_"] for f in features]
        positives = [f["positive_"] for f in features]
        negatives = [f["negative_"] for f in features]
        scores = [f["score_"] for f in features]

        if isinstance(qq[0], list):
            qq = sum(qq, [])
        if isinstance(positives[0], list):
            positives = sum(positives, [])
        if isinstance(negatives[0], list):
            negatives = sum(negatives, [])

        q_collated = self.tokenizer.pad(
            qq,
            padding='max_length',
            max_length=self.max_q_len,
            return_tensors="pt",
        )
        positives_collated = self.tokenizer.pad(
            positives,
            padding='max_length',
            max_length=self.max_p_len,
            return_tensors="pt",
        )
        negatives_collated = self.tokenizer.pad(
            negatives,
            padding='max_length',
            max_length=self.max_p_len,
            return_tensors="pt",
        )
        scores_collated = torch.tensor(scores)

        return q_collated, positives_collated, negatives_collated, scores_collated


@dataclass
class ListwiseDistillationCollator(DataCollatorWithPadding):

    max_q_len: int = 32
    max_p_len: int = 128

    def __call__(self, features):
        qq = [f["query_"] for f in features]
        dd = [f["passages"] for f in features]
        scores = [f["scores_"] for f in features]

        if isinstance(qq[0], list):
            qq = sum(qq, [])
        if isinstance(dd[0], list):
            dd = sum(dd, [])

        q_collated = self.tokenizer.pad(
            qq,
            padding='max_length',
            max_length=self.max_q_len,
            return_tensors="pt",
        )
        d_collated = self.tokenizer.pad(
            dd,
            padding='max_length',
            max_length=self.max_p_len,
            return_tensors="pt",
        )
        scores_collated = torch.tensor(scores)

        return q_collated, d_collated, scores_collated


@dataclass
class DRInferenceCollator(DefaultDataCollator):
    def __call__(self, features):
        text_ids = [x["text_id"] for x in features]
        collated_features = super().__call__(features)
        return text_ids, collated_features


@dataclass
class RRInferenceCollator(DefaultDataCollator):
    def __call__(self, features):
        query_ids = [x["query_id"] for x in features]
        doc_ids = [x["doc_id"] for x in features]
        collated_features = super().__call__(features)
        return query_ids, doc_ids, collated_features

@dataclass
class CQGInferenceCollator(DefaultDataCollator):
    def __call__(self, features):
        query_ids = [x["query_id"] for x in features]
        pos_doc_ids = [x["pos_doc_id"] for x in features]
        neg_doc_ids = [x["neg_doc_id"] for x in features]
        collated_features = super().__call__(features)
        return query_ids, pos_doc_ids, neg_doc_ids, collated_features