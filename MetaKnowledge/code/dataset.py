import json
import os
import copy
from torch.utils.data import Dataset
from dictionary import Dictionary
import torch
import lmdb
import sys
import numpy as np
import networkx as nx
from tqdm import tqdm
import pickle
import random
import pandas as pd

class Seq2SeqDataset(Dataset):
    def __init__(self, data_path="", vocab_file="/root/metakg/data/vocab.txt", device="cpu", args=None):
        self.files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.pkl')]
        random.shuffle(self.files)
        self.current_file_index = 0
        self.current_file = None
        self.current_lines = []
        self.tem_index = 0
        self.load_file()

        self.vocab_file = vocab_file
        self.device = device
        self.seq_len = args.max_len
    
        try:
            self.dictionary = Dictionary.load(vocab_file)
        except FileNotFoundError:
            self.dictionary = Dictionary()
            self._init_vocab()
        self.padding_idx = self.dictionary.pad()
        self.len_vocab = len(self.dictionary)

        self.args = args

    def load_file(self):
        if self.current_file is not None:
            self.current_file.close()
        self.tem_index += len(self.current_lines)
        self.current_file = open(self.files[self.current_file_index], 'rb')
        self.current_lines = pickle.load(self.current_file)
        

    
    def __len__(self):
        return 1480027960



    def _init_vocab(self):
        file_path = './metakg/data/id_name.csv'
        data = pd.read_csv(file_path)
        ids = data['id'].tolist()
        for name in ids:
            self.dictionary.add_symbol(name)
        self.dictionary.save(self.vocab_file)

    def __getitem__(self, index):
        original_index = index  
        try:
            index -= self.tem_index
            if index >= len(self.current_lines):
                self.current_file_index += 1
                if self.current_file_index >= len(self.files):
                    raise IndexError("Index {} is out of the bounds of the dataset".format(original_index))
                self.load_file()
                index -= len(self.current_lines)


            src_line = self.current_lines[index]
            ids = src_line["ids"]
            source_id =torch.IntTensor( ids[:-1])
            target_id = torch.IntTensor(ids[1:])
            type_id =torch.IntTensor(src_line["type_ids"])
            assert len(source_id)==len(target_id)==len(type_id)
            mask = torch.ones_like(target_id)

            return {
                "id": index,
                "tgt_length": len(target_id),
                "source": source_id,
                "target": target_id,
                "type": type_id,
                "mask": mask,
            }
        except IndexError as e:
            print(f"Index error: {original_index}")
            raise e

    def collate_fn(self, samples):
        lens = [sample["tgt_length"] for sample in samples]
        max_len = self.seq_len
        bsz = len(lens)
        source = torch.LongTensor(bsz, max_len)
        type_idx = torch.LongTensor(bsz, max_len)
        mask = torch.zeros(bsz, max_len)
        source.fill_(self.dictionary.pad())
        type_idx.fill_(self.dictionary.null())
        target = copy.deepcopy(source)

        ids = []
        for idx, sample in enumerate(samples):
            ids.append(sample["id"])
            source_ids = sample["source"]
            target_ids = sample["target"]
            type_ids = sample["type"]
            assert len(target_ids)==len(source_ids)

            source[idx, 0:sample["tgt_length"]] = source_ids
            target[idx, 0:sample["tgt_length"]] = target_ids
            type_idx[idx, 0:sample["tgt_length"]] = type_ids
            mask[idx, 0: sample["tgt_length"]] = sample["mask"]

        return {
            "ids": torch.LongTensor(ids).to(self.device),
            "lengths": torch.LongTensor(lens).to(self.device),
            "source": source.to(self.device),
            "target": target.to(self.device),
            "type_id": type_idx.to(self.device),
            "mask": mask.to(self.device),
        }

                
class TestDataset(Dataset):
    def __init__(self, data_path=" ", vocab_file="/root/metakg/data/vocab.txt", device="cpu", src_file=None):

        src_file = os.path.join(data_path, src_file)
        with open(src_file) as f:
            self.src_lines = f.readlines()
        self.vocab_file = vocab_file
        self.device = device

        try:
            self.dictionary = Dictionary.load(vocab_file)
        except FileNotFoundError:
            self.dictionary = Dictionary()
            self._init_vocab()
        self.padding_idx = self.dictionary.pad()
        self.len_vocab = len(self.dictionary)

    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, index):
        src_line = self.src_lines[index].strip().split('\t')
        ids, type_ids = self.dictionary.encode_line(src_line)
        source_id = ids[0]
        target_id = ids[1]
        type_id = type_ids[0]
        mask = torch.ones_like(target_id)
        return {
            "id": index,
            "source": source_id,
            "target": target_id,
            "type": type_id,
            "mask": mask,
        }

    def collate_fn(self, samples):
        bsz = len(samples)
        source = torch.LongTensor(bsz, 1)
        target = torch.LongTensor(bsz, 1)
        type_idx = torch.LongTensor(bsz, 1)

        ids =  []
        for idx, sample in enumerate(samples):
            ids.append(sample["id"])
            source_ids = sample["source"]
            target_ids = sample["target"]
            type_ids = sample["type"]
            source[idx, :] = source_ids
            target[idx, :] = target_ids
            type_idx[idx, :] = type_ids
        
        return {
            "ids": torch.LongTensor(ids).to(self.device),
            "source": source.to(self.device),
            "target": target.to(self.device),
            "type_id": type_idx.to(self.device),
        }