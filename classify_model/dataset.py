import numpy
from numpy import vectorize
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):

  def __init__(self, texts, targets, tokenizer, max_len=512):
    self.texts = texts
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):
    return len(self.texts)

  def __getitem__(self, idx):
    text = str(self.texts[idx])
    target = self.targets[idx]

    encoding = self.tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=self.max_len,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )

    return {
      'text': text,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long)
    }


class SCCL_CustomDataset(Dataset):

  def __init__(self, texts, texts1, targets, tokenizer, max_len=512):
    self.texts = texts
    self.texts1 = texts1
    self.targets = targets 
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):
    return len(self.texts)

  def __getitem__(self, idx):
    text = str(self.texts[idx])
    text1 = str(self.texts1[idx])
    target = self.targets[idx]

    encoding = self.tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=self.max_len,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )
    encoding1 = self.tokenizer.encode_plus(
        text1,
        add_special_tokens=True,
        max_length=self.max_len,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )

    return {
      'text': text,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'text1': text1,
      'input_ids1': encoding1['input_ids'].flatten(),
      'attention_mask1': encoding1['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long)
    }

class TraditionalDataset(Dataset):

  def __init__(self, texts, targets, max_len, embeddings):
    self.texts = texts
    self.targets = targets
    self.max_len = max_len
    self.embeddings = embeddings

  def __len__(self):
    return len(self.texts)

  def __getitem__(self, idx):
    text = str(self.texts[idx])
    target = self.targets[idx]

    vectors = numpy.zeros(shape=(256, 768))
    for i in range(min(len(text.split()), 256)):
        vectors[i] = self.embeddings[text.split()[i]]

    return {
      'text': text,
      'vector': vectors,
      'targets': torch.tensor(target, dtype=torch.long)
    }