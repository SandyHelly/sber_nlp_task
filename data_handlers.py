import random

from torch.utils.data import Dataset, DataLoader, ConcatDataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoConfig, BertModel, XLMRobertaTokenizer
from typing import Optional, Union


class MsMarcoDataset(Dataset):
  def __init__(self, tokenizer_inst: XLMRobertaTokenizer,
               tokenizer_name: str,
               dataset_type: str,
               cache_dir_path: str,
               max_length: int = 128,
               max_samples: Union[int, None] = None):
    self.tokenizer = tokenizer_inst.from_pretrained(tokenizer_name)
    self.max_length = max_length

    dataset = load_dataset('microsoft/ms_marco', 'v1.1',
                           cache_dir=cache_dir_path, split=dataset_type)
    self.query = dataset['query']
    self.doc = dataset['passages']

    if max_samples:
      if len(self.query) > max_samples:
        self.query = self.query [:max_samples]
      if len(self.doc) > max_samples:
        self.doc = self.doc[:max_samples]

  def __len__(self) -> int:
    return len(self.query)

  def __getitem__(self, idx: int) -> tuple:
    q_token_out = self.tokenizer(self.query[idx], max_length=self.max_length, padding='max_length',
                                 truncation=True, return_tensors='pt')
    passages = self.doc[idx]
    try:
      pos_ix = passages['is_selected'].index(1)
    except:
      pos_ix = 0
    neg_ix = random.choice([i for i in range(len(passages['is_selected'])) if i != pos_ix])

    d_token_out = self.tokenizer(passages['passage_text'][pos_ix], max_length=self.max_length, padding='max_length',
                                 truncation=True, return_tensors='pt')

    n_token_out = self.tokenizer(passages['passage_text'][neg_ix], max_length=self.max_length, padding='max_length',
                                truncation=True, return_tensors='pt')

    return q_token_out, d_token_out, n_token_out


class MsMarcoTestDataset(MsMarcoDataset):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def __getitem__(self, idx: int) -> tuple:
    q_token_out = self.tokenizer(self.query[idx], max_length=self.max_length, padding='max_length',
                                 truncation=True, return_tensors='pt')

    passages = self.doc[idx]
    p_list = []

    for passage_text in passages['passage_text']:
      p_token_out = self.tokenizer(passage_text, max_length=self.max_length, padding='max_length',
                                   truncation=True, return_tensors='pt')
      p_list.append(p_token_out)
    return q_token_out, passages['is_selected'], *p_list


class RuEnParaphraseDataset(Dataset):
  def __init__(self, tokenizer_inst: XLMRobertaTokenizer,
               tokenizer_name: str,
               dataset_type: str,
               cache_dir_path: str,
               max_length: int = 128,
               max_samples: Union[int, None] = None):
    self.tokenizer = tokenizer_inst.from_pretrained(tokenizer_name)
    self.max_length = max_length

    dataset = load_dataset('fyaronskiy/ru-paraphrase-NMT-Leipzig-cleaned', cache_dir=cache_dir_path, split=dataset_type)
    self.query = dataset['en']
    self.doc = dataset['ru']

    if max_samples:
      if len(self.query) > max_samples:
        self.query = self.query [:max_samples]
      if len(self.doc) > max_samples:
        self.doc = self.doc[:max_samples]

  def __len__(self) -> int:
    return len(self.query)

  def __getitem__(self, idx) -> tuple:
    q_token_out = self.tokenizer(self.query[idx], max_length=self.max_length, padding='max_length',
                                 truncation=True, return_tensors='pt')

    neg_ix = random.choice([i for i in range(len(self.query)) if i != idx])

    d_token_out = self.tokenizer(self.doc[idx], max_length=self.max_length, padding='max_length',
                                 truncation=True, return_tensors='pt')

    n_token_out = self.tokenizer(self.doc[neg_ix], max_length=self.max_length, padding='max_length',
                                 truncation=True, return_tensors='pt')

    return q_token_out, d_token_out, n_token_out
