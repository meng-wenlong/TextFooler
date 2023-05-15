import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler, TensorDataset

from transformers import RobertaForSequenceClassification, RobertaTokenizer

import os


class HFInputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask):
        self.input_ids = input_ids
        self.input_mask = input_mask


class Dataset_RoBERTa(Dataset):
    def __init__(self,
                 pretrained_dir,
                 max_seq_length=128,
                 batch_size=32):
        self.tokenizer = RobertaTokenizer.from_pretrained(os.path.join(pretrained_dir, 'tokenizer'))
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

    def convert_examples_to_features(self, examples):
        features = []
        for (ex_index, text_a) in enumerate(examples):
            sent_a = ' '.join(text_a)
            
            encoding = self.tokenizer(
                sent_a,
                max_length=self.max_seq_length,
                padding="max_length",
                truncation=True
            )

            input_ids = encoding['input_ids']
            input_mask = encoding['attention_mask']

            features.append(
                HFInputFeatures(input_ids=input_ids,
                              input_mask=input_mask)
            )
        return features

    def transform_text(self, data, batch_size=32):
        # transform data into seq of embeddings
        eval_features = self.convert_examples_to_features(data)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)

        return eval_dataloader



class Infer_RoBERTa(nn.Module):
    def __init__(self,
                 pretrained_dir,
                 nclasses,
                 max_seq_length=128,
                 batch_size=32):
        super(Infer_RoBERTa, self).__init__()
        self.model = RobertaForSequenceClassification.from_pretrained(pretrained_dir, num_labels=nclasses).cuda()

        # construct dataset loader
        self.dataset = Dataset_RoBERTa(pretrained_dir, max_seq_length=max_seq_length, batch_size=batch_size)

    def text_pred(self, text_data, batch_size=32):
        self.model.eval()

        dataLoader = self.dataset.transform_text(text_data, batch_size=batch_size)

        probs_all = []

        for input_ids, input_mask in dataLoader:
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()

            with torch.no_grad():
                logits = self.model(input_ids, input_mask)['logits']
                probs = nn.functional.softmax(logits, dim=-1)
                probs_all.append(probs)

        return torch.cat(probs_all, dim=0)

        