import pickle as pickle
import os
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit

os.chdir('./KorEDA/')
from KorEDA.eda import *
os.chdir('../')

class RE_Dataset(torch.utils.data.Dataset):
  """ Dataset 구성을 위한 class."""
  def __init__(self, pair_dataset, labels):
    self.pair_dataset = pair_dataset
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)

def preprocessing_dataset(dataset):
  """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
  subject_entity = []
  object_entity = []
  good_flag = False
  bad_flag = False
  for i,j in zip(dataset['subject_entity'], dataset['object_entity']):


    try:
      if not good_flag:
        good_flag = True
        print("good case")
        print(i)
        print(j)
        print(type(i))
      i = i[1:-1].split(',')[0].split(':')[1]
      j = j[1:-1].split(',')[0].split(':')[1]

      subject_entity.append(i)
      object_entity.append(j)
    except:
      if not bad_flag:
        print("bad case")
        bad_flag = True
        print(i)
        print(j)
        print(type(i))
  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
  duplied = out_dataset[out_dataset.duplicated(subset=['sentence','subject_entity','object_entity'])]
  duplied_no_idx = duplied[duplied['label'] == 'no_relation']['id'].to_list()
  for idx in duplied_no_idx:
    out_dataset.drop(out_dataset.loc[out_dataset['id']==idx].index, inplace=True)
  out_dataset = out_dataset.drop_duplicates(subset=['sentence','subject_entity','object_entity','label'],keep='first')  
  return out_dataset

def load_data(dataset_dir):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)
  dataset = preprocessing_dataset(pd_dataset)
  
  return dataset

def tokenized_dataset(dataset, tokenizer):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  concat_entity = []
  for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
    temp = ''
    temp = e01 + '[SEP]' + e02
    concat_entity.append(temp)
  tokenized_sentences = tokenizer(
      concat_entity,
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True,
      return_token_type_ids=False,
      )
  return tokenized_sentences



def augmentation(target_set):
  new_row = []
  for idx, sentence in enumerate(target_set.sentence):
    words = sentence.split(' ')
    words = [word for word in words if word is not ""]

    eg = target_set.iloc[idx]

    sub = eval(eg['subject_entity'])
    sub_ent = sub['word']
    sub_len = sub['end_idx'] - sub['start_idx']


    obj = eval(eg['object_entity'])
    obj_ent = obj['word']
    obj_len = obj['end_idx'] - obj['start_idx']

    times = 0
    
    trial = 0
    num_failed = 0
    while True:
      res = random_swap(words, 5)
      res = " ".join(res)
      trial += 1
      if trial == 10:
        num_failed += 1
        break

      if res.find(sub_ent) == -1 or res.find(obj_ent) == -1:
        continue

      _sub = sub.copy()
      _sub['start_idx'] = res.index(sub_ent)
      _sub['end_idx'] = _sub['start_idx'] + sub_len

      _obj = obj.copy()
      _obj['start_idx'] = res.index(obj_ent)
      _obj['end_idx'] = _obj['start_idx'] + obj_len

      new_row.append( [ len(target_set) + times + 1, res, str(_sub), str(_obj), eg['label'], eg['source'] ] )
      times += 1
      
      if times == 10:
        break
  return new_row



def load_stratified_data(dataset_dir, aug = None):
  """ csv 파일을 경로에 맡게 불러 옵니다. """  
  split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
  pd_dataset = pd.read_csv(dataset_dir)  
  
  for train_index, test_index in split.split(pd_dataset, pd_dataset["label"]):
    strat_train_set = pd_dataset.loc[train_index]
    strat_dev_set = pd_dataset.loc[test_index]

  if aug:
    before = len(strat_train_set)
    print("processing data augmentation... may take a while...")

    for l in [ "per:children", "per:colleagues", "per:other_family" ]:
        target_set = strat_train_set[strat_train_set.label == l]
        new_rows = augmentation(target_set)
        new_df = pd.DataFrame(new_rows, columns = target_set.columns)
        new_df
        strat_train_set = pd.concat([strat_train_set, new_df])

    after = len(strat_train_set)    
    print("augmentation finishied. Before aug size: %d, After aug size: %d" %(before, after))

  train_dataset = preprocessing_dataset(strat_train_set)  
  dev_dataset = preprocessing_dataset(strat_dev_set)  
  return train_dataset, dev_dataset

